#define PCL_NO_PRECOMPILE
#include <kontiki/trajectory_estimator.h>
#include <kontiki/trajectories/split_trajectory.h>
#include <kontiki/trajectories/uniform_r3_spline_trajectory.h>
#include <kontiki/measurements/position_measurement.h>
#include <kontiki/measurements/lidar_point2point_measurement.h>
#include <kontiki/measurements/lidar_point2plane_measurement.h>
#include <kontiki/sensors/vlp16_lidar.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <Eigen/Core>
#include <ceres/ceres.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/flann_search.hpp>

#include <fstream>
#include <memory>
#include <vector>
#include <math.h>
#include <time.h>

#include "utils/TicToc.h"

// 自定义点
struct PointXYZIT {
  PCL_ADD_POINT4D
  float intensity;  // 存相对时间
  double timestamp; // 存绝对时间 = 点云时间戳 + 相对时间
  int id; // 点序号
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (double, timestamp, timestamp)
								  (int, id, id))

typedef PointXYZIT PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> PointCloudIntensity;

using LiDARSensor = kontiki::sensors::VLP16LiDAR;
using R3TrajEstimator         = kontiki::TrajectoryEstimator<kontiki::trajectories::UniformR3SplineTrajectory>;
using PositionMeasurement     = kontiki::measurements::PositionMeasurement;
using Result                  = std::unique_ptr<kontiki::trajectories::TrajectoryEvaluation<double>>;
using Point2PointMeasurement         = kontiki::measurements::LiDARPoint2PointMeasurement<LiDARSensor>;
using Point2PlaneMeasurement         = kontiki::measurements::LiDARPoint2PlaneMeasurement<LiDARSensor>;

struct EncodeData {
	double timestamp;
	double angle;  // 使用弧度制
};

// 点云匹配对
struct MatchPair {
	PointType *pointa;
	PointType *pointb;
	PointType *pointc;
	PointType *pointCur;
	MatchPair() : pointa(nullptr),  pointb(nullptr),  pointc(nullptr), pointCur(nullptr) {}
};

const double PI = 3.1415926;
double timeDelayMax = 0.5;  //lidar和码盘之间最大的时间延迟
double timeDelay = 0.0; // lidar和码盘之间的时间延迟
double dataStartTime = 0.0;
double dataEndTime = 0.0;
double angleDist1 = 170.0; // 构建地图时的码盘角度差
double angleDist2 = 190.0;

std::vector<PointCloud::Ptr> lidarData;
std::vector<int> cloudIDs;
std::vector<int> surfcloudIDs;
std::vector<int> cornorcloudIDs;
std::vector<MatchPair> matchPairs;
std::vector<double> lidarTimestamps;
std::vector<EncodeData> encodeData;

Eigen::Vector3d exTransposeLidar2Encode;
Eigen::Matrix3d exOrientationLidar2Encode;

ros::Publisher pubLocalMap;
ros::Publisher pubLocalSurf;


double deg2rad(double deg) {
	return deg / 180.0 * PI;
}

double rad2deg(double rad) {
	return rad / PI * 180.0;
}

inline double lidarTime2EncodeTime(double lidarTime) {
	return lidarTime + timeDelay;
}

template <typename PointT>
inline void PublishCloudMsg(ros::Publisher& publisher,
                            const pcl::PointCloud<PointT>& cloud,
                            const ros::Time& stamp,
                            std::string frame_id) {
	sensor_msgs::PointCloud2 msg;
	pcl::toROSMsg(cloud, msg);
	msg.header.stamp = stamp;
	msg.header.frame_id = frame_id;
	publisher.publish(msg);
}

double normalTo2pi(double angle) {
	const double dpi = 2.0 * PI;
	while (angle > dpi) {
		angle -= dpi;
	}
	return angle;
}

void IntensityCloud2CustomCloud(PointCloudIntensity &ci, PointCloud &co) {
	int csize = ci.points.size();
	co.points.resize(csize);
	for (int i = 0; i < csize; ++i) {
		PointI &pi = ci.points[i];
		PointType &po = co.points[i];
		po.x = pi.x;
		po.y = pi.y;
		po.z = pi.z;
		po.intensity = 0.0;
		po.timestamp = 0.0;
		po.id = 0;
	}
}

bool readDataBag(const std::string path,
		const std::string encode_topic,
		const std::string lidar_topic,
		const double bag_start = -1.0,
		const double bag_durr = -1.0) {

	std::shared_ptr<rosbag::Bag> bag_;
	bag_.reset(new rosbag::Bag);
	bag_->open(path, rosbag::bagmode::Read);

	rosbag::View view;
	{
		std::vector<std::string> topics;
		topics.push_back(encode_topic);
		topics.push_back(lidar_topic);

		rosbag::View view_full;
		view_full.addQuery(*bag_);
		ros::Time time_init = view_full.getBeginTime();
		time_init += ros::Duration(bag_start);
		ros::Time time_finish = (bag_durr < 0)?
								view_full.getEndTime() : time_init + ros::Duration(bag_durr);
		view.addQuery(*bag_, rosbag::TopicQuery(topics), time_init, time_finish);
	}

	for (rosbag::MessageInstance const m : view) {
		const std::string &topic = m.getTopic();

		if (lidar_topic == topic) {
			PointCloud::Ptr pointcloud = pcl::make_shared<PointCloud>();
			PointCloudIntensity::Ptr pcIntensity = pcl::make_shared<PointCloudIntensity>();
			double timestamp = 0;

			if (m.getDataType() == std::string("velodyne_msgs/VelodyneScan")) {
				// velodyne_msgs::VelodyneScan::ConstPtr vlp_msg =
				// 		m.instantiate<velodyne_msgs::VelodyneScan>();
				// timestamp = vlp_msg->header.stamp.toSec();
				// p_LidarConvert_->unpack_scan(vlp_msg, pointcloud);
			}

			if (m.getDataType() == std::string("sensor_msgs/PointCloud2")) {
				sensor_msgs::PointCloud2::ConstPtr scan_msg =
						m.instantiate<sensor_msgs::PointCloud2>();
				timestamp = scan_msg->header.stamp.toSec();
				pcl::fromROSMsg(*scan_msg, *pcIntensity);
				IntensityCloud2CustomCloud(*pcIntensity, *pointcloud);
			}

			lidarData.emplace_back(pointcloud);
			lidarTimestamps.emplace_back(timestamp);
		}

		if (encode_topic == topic) {
			static int n_count = 0;  //绝对边界计数
			static double last_angle_tmp = 500.0;
			static int circle_count = 0;
			static double last_encode_angle = 0.0;

			sensor_msgs::JointState::ConstPtr angleIn = m.instantiate<sensor_msgs::JointState>();
			if (angleIn.get() == nullptr) {
				std::cout << "angleIn.get() == nullptr\n";
			}
			
			double time = angleIn->header.stamp.toSec();

			if(angleIn->position.size()==0)
				continue;
			if(last_angle_tmp > 130.0)  //第一次
				last_angle_tmp = angleIn->position[0];
			
			// 绝对角度：angle = (angle_in + n_count*2560)
			
			// *******检测是否越过分界线 ****************
			if(angleIn->position[0]>120.0 && last_angle_tmp<-120.0)
				n_count -= 1;  //回退了一圈
			else if(angleIn->position[0]<-120.0 && last_angle_tmp>120.0)
				n_count += 1;  //前进了一圈
			
			// 第一种方式，不用担心n_count过大 导致越界
			double tmp = angleIn->position[0] * 10.0;
			int int1 = (int)(tmp/360.0);
			n_count = n_count % 9;
			tmp = tmp - int1*360.0 + 40.0*n_count;  //初步归一化
			tmp = tmp - (int)(tmp/360.0) * 360.0;  //不满360的部分(0 ~ 359.999)
			if(tmp<0.0) tmp = 360.0 + tmp;
			
			// 第二种方式直接点，其实这种方式能用很久 不用担心
			/*double tmp = angleIn->position[0] * 10.0 + n_count * 2560.0;
			int int1 = (int)(tmp/360.0);
			tmp = tmp - int1*360.0 ;
			if(tmp<0.0) tmp = 360.0 + tmp;*/

			last_angle_tmp = angleIn->position[0];
			if (last_encode_angle <= 360.0 && last_encode_angle >= 320.0 && tmp >= 0.0 && tmp <= 40) {
				circle_count++;
			}
			last_encode_angle = tmp;
			tmp = circle_count * 360.0 + tmp;

			encodeData.emplace_back();
			encodeData.back().timestamp = time;
			encodeData.back().angle = deg2rad(tmp);
		}
	}

	std::cout << lidar_topic << ": " << lidarData.size() << std::endl;
	std::cout << encode_topic << ": " << encodeData.size() << std::endl << std::endl;
	return true;
}

void adjustDataset() {
    assert(encodeData.size() > 0 && "No encode data. Check your bag and encode topic");
    assert(lidarData.size() > 0 && "No scan data. Check your bag and lidar topic");

    assert(lidarTimestamps.front() < encodeData.back().timestamp
           && lidarTimestamps.back() > encodeData.front().timestamp
           && "Unvalid dataset. Check your dataset.. ");

	// 点云数据 “夹在” 码盘数据中间
	/*
	encode:  |  |  |  |  |  |  |  | …… |  |  |  |  |  |  |  |
	lidar:     |           |        ……   |             |
	*/
    if (lidarTimestamps.front() - timeDelayMax >= encodeData.front().timestamp) {
		// 说明此时lidar数据是“夹在”码盘数据中的
		std::cout << "front is ok\n";
    } else {
		// 去掉前面的lidar数据，以使数据满足要求
		while (lidarTimestamps.front() - timeDelayMax < encodeData.front().timestamp) {
			lidarTimestamps.erase(lidarTimestamps.begin());
			lidarData.erase(lidarData.begin());
			std::cout << "pop front lidar\n";
		}
		std::cout << "after pop front is ok\n";
    }

	if (lidarTimestamps.back() + timeDelayMax <= encodeData.back().timestamp) {
		// 说明此时lidar数据是“夹在”码盘数据中的
		std::cout << "back is ok\n";
	} else {
		// 去掉后面的lidar数据，以使数据满足要求
		while (lidarTimestamps.back() + timeDelayMax > encodeData.back().timestamp) {
			lidarTimestamps.pop_back();
			lidarData.pop_back();
			std::cout << "pop back lidar\n";
		}
		std::cout << "after pop back is ok\n";
	}

	dataStartTime = encodeData.front().timestamp;
	dataEndTime = encodeData.back().timestamp;
    std::cout << "lidar.front.timestamp - encode.front.timestamp = " << lidarTimestamps.front() - encodeData.front().timestamp
	<< "\nencode.back.timestamp - lidar.back.timestamp = " << encodeData.back().timestamp - lidarTimestamps.back() << "\n\n";
}

template<typename T>
void averageTimeDownSmaple(T &pci, T &pco, int step) {
	for (int idx = 0; idx < pci.points.size(); idx+= step) {
    	pco.push_back(pci.points.at(idx));
  	}
}

void extracFeatureAndCalculateTime(PointCloud::Ptr &laserCloudIn,
								   const double cloudTimestamp,
								   PointCloud::Ptr &cornerCloud, 
								   PointCloud::Ptr &surfCloud) {
	
	PointCloud::Ptr laserCloudScans[16];
	for (int i = 0; i < 16; i++) {
		laserCloudScans[i].reset(new PointCloud());
	}
	PointCloud::Ptr laserCloud               = pcl::make_shared<PointCloud>();  //原始点云
	PointCloud::Ptr cornerPointsSharp        = pcl::make_shared<PointCloud>();
	PointCloud::Ptr cornerPointsLessSharp    = pcl::make_shared<PointCloud>();
	PointCloud::Ptr surfPointsFlat           = pcl::make_shared<PointCloud>();
	PointCloud::Ptr surfPointsLessFlat       = pcl::make_shared<PointCloud>();
	PointCloud::Ptr surfPointsLessFlatScan   = pcl::make_shared<PointCloud>();  // 用于提取次面点的中间量
	PointCloud::Ptr surfPointsLessFlatScanDS = pcl::make_shared<PointCloud>();
	int scanStartInd[16];
	int scanEndInd[16];
	float cloudCurvature[40000];
	int cloudSortInd[40000];
	int cloudNeighborPicked[40000];
	int cloudLabel[40000];

	//去除无效点
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*laserCloudIn,*laserCloudIn, indices);
	indices.clear();
	int cloudSize = laserCloudIn->points.size();
	
	//进来点云的起始角度和终止角度
	float startOri = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
	float endOri = -atan2(laserCloudIn->points[cloudSize - 1].y,
						 laserCloudIn->points[cloudSize - 1].x) + 2 * PI;
	//把角度限定在pi-2pi之内
	if (endOri - startOri > 3 * PI) {
		endOri -= 2 * PI;
	} else if (endOri - startOri < PI) {
		endOri += 2 * PI;
	}

	bool halfPassed = false;
	int count = cloudSize;
	PointType point;
	for (int i = 0; i < cloudSize; i++) {
		point = laserCloudIn->points[i];
		
		//滤除有效距离外的点
		float dis2orig2 = point.x * point.x + point.y * point.y +point.z * point.z;
		if (dis2orig2 < 0.04 || dis2orig2 > 4900.0) {
			count--;
			continue;
		}
	
		//计算当前点属于哪条线，范围 0-15
		float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180.0 / PI;
		int scanID;
		int angle2 = int(angle + (angle < 0.0 ? -0.5 : +0.5));
		if (angle2 > 0) {
			scanID = angle2/2 + 8;
		} else {
			scanID = angle2/2 + 7;
		}

		if (scanID > 15) {
			count--;
			continue;
		}
		
		//当前点位于一条扫描线的什么角度
		float ori = -atan2(point.y, point.x);
		if (!halfPassed) {
			if (ori < startOri - PI / 2) {
				ori += 2 * PI;
			} else if (ori > startOri + PI * 3 / 2) {
				ori -= 2 * PI;
			}

			if (ori - startOri > PI){
				halfPassed = true;
			}
		} else {
			ori += 2 * PI;
			if (ori < endOri - PI * 3 / 2) {
				ori += 2 * PI;
			}
			else if (ori > endOri + PI / 2) {
				ori -= 2 * PI;
			}
		}
		
		//intensity被改写了，这个值能确定该点在什么位置，第几条线的什么角度
		float relTime = (ori - startOri) / (endOri - startOri);
		point.intensity = 0.1 * relTime;
		bool validPoint = true;
		if (point.intensity < 0) {
			// std::cout << "relative time < 0\n";
			point.intensity = 0.0;
			validPoint = false;
		} else if (point.intensity > 0.1) {
			point.intensity = 0.1;
			// std::cout << "relative time > 0.1\n";
			validPoint = false;
		}
		point.timestamp = cloudTimestamp + point.intensity;
		if (validPoint) {
			laserCloudScans[scanID]->push_back(point);
		}
	}

	//整理点云，准备提取特征点
    for (int i = 0; i < 16; i++) {
		if(laserCloudScans[i]->points.size() < 100) {
			scanStartInd[i] = 0;
			scanEndInd[i] = 0;
			continue;
		}
		
		scanStartInd[i] = laserCloud->points.size() + 5;  //i扫描线在一维数组中的起点
		*laserCloud += *laserCloudScans[i];  //放到一个数组中
		scanEndInd[i] = laserCloud->points.size() - 5;  //i扫描线在一维数组中的终点
    }
	cloudSize = laserCloud->size();
	
	//开始计算特征点
	//计算c值 s表示i点周围10个点，前5后5
	int scanCount = -1;
	for (int i = 5; i < cloudSize - 5; i++) {

		float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x
				 + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x
				 + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x
				 + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
				 + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
				 + laserCloud->points[i + 5].x;
		float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y
				 + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y
				 + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y
				 + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
				 + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
				 + laserCloud->points[i + 5].y;
		float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z
				 + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z
				 + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z
				 + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
				 + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
				 + laserCloud->points[i + 5].z;

		cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;//点云曲率
		cloudSortInd[i] = i;//排序
		cloudNeighborPicked[i] = 0;//
		cloudLabel[i] = 0;
	}

	//消除遮挡等
	for (int i = 5; i < cloudSize - 6; i++) {
		float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
		float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
		float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
		float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

		if (diff > 0.1) {
			//用于消去拐点.特征为遮挡的情况
			//点的深度

			float depth12 = laserCloud->points[i].x * laserCloud->points[i].x +
								laserCloud->points[i].y * laserCloud->points[i].y +
								laserCloud->points[i].z * laserCloud->points[i].z;

			float depth22 = laserCloud->points[i + 1].x * laserCloud->points[i + 1].x +
								laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
								laserCloud->points[i + 1].z * laserCloud->points[i + 1].z;

			if (depth12 > depth22) {
				float bilv = sqrt(depth22 / depth12);
				diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * bilv;
				diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * bilv;
				diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * bilv;

				if ((diffX * diffX + diffY * diffY + diffZ * diffZ) < 0.01 * depth22) {
					cloudNeighborPicked[i - 5] = 1;
					cloudNeighborPicked[i - 4] = 1;
					cloudNeighborPicked[i - 3] = 1;
					cloudNeighborPicked[i - 2] = 1;
					cloudNeighborPicked[i - 1] = 1;
					cloudNeighborPicked[i] = 1;
				}
			} else {
				float bilv = sqrt(depth12 / depth22);
				diffX = laserCloud->points[i + 1].x * bilv - laserCloud->points[i].x;
				diffY = laserCloud->points[i + 1].y * bilv - laserCloud->points[i].y;
				diffZ = laserCloud->points[i + 1].z * bilv - laserCloud->points[i].z;

				if ((diffX * diffX + diffY * diffY + diffZ * diffZ) < 0.01 * depth12) {
					cloudNeighborPicked[i + 1] = 1;
					cloudNeighborPicked[i + 2] = 1;
					cloudNeighborPicked[i + 3] = 1;
					cloudNeighborPicked[i + 4] = 1;
					cloudNeighborPicked[i + 5] = 1;
					cloudNeighborPicked[i + 6] = 1;
				}
			}
		}

		float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
		float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
		float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
		float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

		float dis = laserCloud->points[i].x * laserCloud->points[i].x
			   + laserCloud->points[i].y * laserCloud->points[i].y
			   + laserCloud->points[i].z * laserCloud->points[i].z;
		
		//消除斜率太大的情况
		if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
			cloudNeighborPicked[i] = 1;
		}
	}

	//=============== 原来的特征提取 ==========
	for (int i = 0; i < 16; i++) {
		surfPointsLessFlatScan->clear();
		//将一条线分为6段，每一段都找到了两个最大的角点和四个最小的平面
		for (int j = 0; j < 6; j++) {
			int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
			int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;
			//升序排序
			for (int k = sp + 1; k <= ep; k++) {
				for (int l = k; l >= sp + 1; l--) {
					if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
						int temp = cloudSortInd[l - 1];
						cloudSortInd[l - 1] = cloudSortInd[l];
						cloudSortInd[l] = temp;
					}
				}
			}
       
			//选取角点
			int largestPickedNum = 0;
			for (int k = ep; k >= sp; k--) {
				int ind = cloudSortInd[k];
				if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1) {
					largestPickedNum++;
					if (largestPickedNum <= 2) {
						cloudLabel[ind] = 2; //点云标志，定义角点、面点。2-最角点，1-次角点
						cornerPointsSharp->push_back(laserCloud->points[ind]);
						cornerPointsLessSharp->push_back(laserCloud->points[ind]);
					} else if (largestPickedNum <= 20) {
						cloudLabel[ind] = 1;//次角点
						cornerPointsLessSharp->push_back(laserCloud->points[ind]);
					} else {
						break;
					}

					cloudNeighborPicked[ind] = 1;
					for (int l = 1; l <= 5; l++) {
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l - 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l - 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l - 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
					for (int l = -1; l >= -5; l--) {
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l + 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l + 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l + 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}

						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}
			
			
			//选取平面点
			int smallestPickedNum = 0;
			for (int k = sp; k <= ep; k++) {
				int ind = cloudSortInd[k];
				if (cloudNeighborPicked[ind] == 0 &&cloudCurvature[ind] < 0.1) {
					cloudLabel[ind] = -1;//面性点的标志 -1
					surfPointsFlat->push_back(laserCloud->points[ind]);
					smallestPickedNum++;
					if (smallestPickedNum >= 4) {
						//选取4个
						break;
					}

					cloudNeighborPicked[ind] = 1;
					for (int l = 1; l <= 5; l++) {
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l - 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l - 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l - 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}

						cloudNeighborPicked[ind + l] = 1;
					}
					for (int l = -1; l >= -5; l--) {
						if((ind+l) < 0)
							continue;
						
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l + 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l + 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l + 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}

						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}

			for (int k = sp; k <= ep; k++) {
				if (cloudLabel[k] <= 0) {
					surfPointsLessFlatScan->push_back(laserCloud->points[k]);
				}
			}
		}

		surfPointsLessFlatScanDS->clear();
		// pcl::VoxelGrid<PointType> downSizeFilter;
		// downSizeFilter.setInputCloud(surfPointsLessFlatScan);
		// downSizeFilter.setLeafSize(0.1, 0.1, 0.1);  //平面点降采样
		// downSizeFilter.filter(*surfPointsLessFlatScanDS);
		averageTimeDownSmaple(*surfPointsLessFlatScan, *surfPointsLessFlatScanDS, 10);

		*surfPointsLessFlat += *surfPointsLessFlatScanDS;
	}
	
	laserCloudIn = laserCloud;
	cornerCloud  = cornerPointsLessSharp;
	surfCloud    = surfPointsLessFlat;
}

void initTrajectory(std::shared_ptr<kontiki::trajectories::SplitTrajectory> &traj, double knot_distance) {
	double traj_start_time = encodeData[0].timestamp;
    double traj_end_time = encodeData.back().timestamp;
    traj = std::make_shared<kontiki::trajectories::SplitTrajectory>
            (knot_distance, knot_distance, traj_start_time, traj_start_time);

    Eigen::Vector3d p0(0,0,0);
    Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity();
    traj->R3Spline()->ExtendTo (traj_end_time, p0);
    traj->SO3Spline()->ExtendTo(traj_end_time, q0);
    std::cout << "spline min time = " << traj->R3Spline()->MinTime() << "  max time = " << traj->R3Spline()->MaxTime() << std::endl;
    std::cout << "spline contral point num = " << traj->R3Spline()->NumKnots() << std::endl;   
}

void optimizeEncodeAngles(std::shared_ptr<kontiki::trajectories::SplitTrajectory> &traj) {
	std::shared_ptr<R3TrajEstimator> estimator_split;  // 用于优化
	estimator_split = std::make_shared<R3TrajEstimator>(traj->R3Spline());
	std::vector< std::shared_ptr<PositionMeasurement>>  msg_list_; // 保存测量值，需要保存智能指针

	for (auto da : encodeData) {
		double time = da.timestamp;
		double value = da.angle;
		Eigen::Vector3d p0(value, 0, 0);
		double weight = 1.0;
		auto m_p0_= std::make_shared<PositionMeasurement>(time, p0, weight);
		msg_list_.push_back(m_p0_);
		estimator_split->template AddMeasurement<PositionMeasurement>(m_p0_);
	}    

	// 求解
	ceres::Solver::Summary summary = estimator_split->Solve(50, false);
	std::cout << summary.BriefReport() << std::endl;
}

Eigen::Matrix3d pr2R(double p, double r) {
	Eigen::Matrix3d Ry;
	Ry << cos(p), 0., sin(p),
			0., 1., 0.,
			-sin(p), 0., cos(p);

	Eigen::Matrix3d Rx;
	Rx << 1., 0., 0.,
			0., cos(r), -sin(r),
			0., sin(r), cos(r);

	return Ry * Rx;
}

bool evaluate(std::shared_ptr<kontiki::trajectories::SplitTrajectory> &traj,
			  double t, Eigen::Vector3d &p) {
    if (traj->MinTime() > t || traj->MaxTime() <= t)
        return false;
    Result result = traj->Evaluate(t, kontiki::trajectories::EvalPosition);
    p = result->position;
    return true;
}

void point2InitEncode(const PointType &pi, PointI &po,
					  const Eigen::Vector3d &trans, const Eigen::Matrix3d &orien, double yaw) {
	
	Eigen::Vector3d vec;
	vec.x() = static_cast<double>(pi.x);
	vec.y() = static_cast<double>(pi.y);
	vec.z() = static_cast<double>(pi.z);
	vec = orien * vec + trans;
	Eigen::Matrix3d orien2init;
	orien2init << cos(yaw), -sin(yaw), 0.,
				  sin(yaw),  cos(yaw), 0.,
				  0.,        0.,       1.;

	vec = orien2init * vec;

	po.x = static_cast<float>(vec.x());
	po.y = static_cast<float>(vec.y());
	po.z = static_cast<float>(vec.z());
	po.intensity = rad2deg(yaw);
}

void point2InitEncode(const PointType &pi, PointType &po,
					  const Eigen::Vector3d &trans, const Eigen::Matrix3d &orien, double yaw) {
	
	Eigen::Vector3d vec;
	vec.x() = static_cast<double>(pi.x);
	vec.y() = static_cast<double>(pi.y);
	vec.z() = static_cast<double>(pi.z);
	vec = orien * vec + trans;
	Eigen::Matrix3d orien2init;
	orien2init << cos(yaw), -sin(yaw), 0.,
				  sin(yaw),  cos(yaw), 0.,
				  0.,        0.,       1.;

	vec = orien2init * vec;

	po.x = static_cast<float>(vec.x());
	po.y = static_cast<float>(vec.y());
	po.z = static_cast<float>(vec.z());
	po.intensity = pi.intensity;
	po.timestamp = pi.timestamp;
	po.id = pi.id;
}

void transformCloud2Init(PointCloud &ci, PointCloudIntensity &co,
						 const Eigen::Vector3d &trans, const Eigen::Matrix3d &orien,
						 std::shared_ptr<kontiki::trajectories::SplitTrajectory> &traj) {
	co.reserve(ci.points.size());
	for (int i = 0; i < ci.points.size(); ++i) {
		PointI po;
		PointType &pi = ci.points[i];
		Eigen::Vector3d pos;
		double ptime = lidarTime2EncodeTime(pi.timestamp);
		if (evaluate(traj, ptime, pos)) {
			double angle = pos.x();
			point2InitEncode(pi, po, trans, orien, -angle);  // 此处取负角度值，因为码盘按顺时针旋转，在右手系中是负角度方向
			co.push_back(po);
		}
	}
}

PointCloud::Ptr transformCloud2Init(PointCloud &ci, const Eigen::Vector3d &trans, const Eigen::Matrix3d &orien,
						 std::shared_ptr<kontiki::trajectories::SplitTrajectory> &traj) {
	PointCloud::Ptr co = pcl::make_shared<PointCloud>();
	co->reserve(ci.points.size());
	for (int i = 0; i < ci.points.size(); ++i) {
		PointType po;
		PointType &pi = ci.points[i];
		Eigen::Vector3d pos;
		double ptime = lidarTime2EncodeTime(pi.timestamp);
		if (evaluate(traj, ptime, pos)) {
			double angle = pos.x();
			point2InitEncode(pi, po, trans, orien, -angle);  // 此处取负角度值，因为码盘按顺时针旋转，在右手系中是负角度方向
			co->push_back(po);
		}
	}
	return co;
}

void calculateCloudIDs(std::vector<pcl::shared_ptr<PointCloud>> &Clouds, std::vector<int> &IDs) {
	int frameSize = Clouds.size();
	IDs.resize(frameSize);
	int allSize = 0;
	for (int i = 0; i < frameSize; ++i) {
		int aFrame = Clouds[i]->points.size();
		for (int j = 0; j < aFrame; ++j) {
			Clouds[i]->points[j].id = allSize + j;
		}
		allSize += aFrame;
		IDs[i] = allSize;
	}
}

bool getFrameAndIndex(int id, int &fid, int &pid, std::vector<int> &IDs) {
	auto it = std::upper_bound(IDs.begin(), IDs.end(), id);
	if (it == IDs.end()) {
		std::cout << "can't get frame and index\n";
		return false;
	}
	fid = it - IDs.begin();
	if (fid == 0) {
		pid = id;
	} else {
		pid = id - IDs[fid - 1];
	}
	return true;
}

void findCorressponds(std::vector<pcl::shared_ptr<PointCloud>> &surfs,
					  std::shared_ptr<kontiki::trajectories::SplitTrajectory> &traj,
					  int stID, int endID, int skip) {
	TicToc tictoc;
	tictoc.Tic();
	int count = 0;
	for (int i = stID; i <= endID; i = i + skip + 1) {
		// 构建地图
		Eigen::Vector3d pos;
		double curTime = lidarTime2EncodeTime(lidarTimestamps[i]);
		evaluate(traj, curTime, pos);
		double curAngle = pos.x();

		PointCloud::Ptr localMap = pcl::make_shared<PointCloud>();
		for (int j = i - 1; j >= 0; --j) {
			// 将码盘角度差大于 170 度的点云构建地图
			double tmpTime = lidarTime2EncodeTime(lidarTimestamps[j]);
			evaluate(traj, tmpTime, pos);
			double tmpAngle = pos.x();
			bool mapValid = (normalTo2pi(curAngle - tmpAngle) > angleDist1) && (normalTo2pi(curAngle - tmpAngle) < angleDist2);
			if (mapValid) {
				*localMap += *transformCloud2Init(*surfs[j], exTransposeLidar2Encode, exOrientationLidar2Encode, traj);
				// std::cout << "surfs [ " << j << " ] size : " << surfs[j]->points.size() << std::endl;
				// std::cout << "localMap size : " << localMap->points.size() << std::endl;
				double delta = curAngle - tmpAngle;
				// 第一圈地图的角度差是[angleDist1, angleDist2],第二圈是[angleDist1 + 360, angleDist2 + 360],第三圈是[angleDist1 + 720, angleDist2 + 720]
				// 因此大于720度的话，可以采集两圈地图
				if (delta > deg2rad(720.0)) break; 
			}
		}

		if (localMap->points.size() < 100) {
			continue;
		}
		count++;
		// char ch;
		// std::cout << "input a to continue\n";
		// while (std::cin >> ch) {
		// 	if (ch == 'a') break;
		// }
		PublishCloudMsg(pubLocalMap, *localMap, ros::Time().fromSec(lidarTimestamps[i]), "motor");
		PointCloud::Ptr surfLocal = transformCloud2Init(*surfs[i], exTransposeLidar2Encode, exOrientationLidar2Encode, traj);
		PublishCloudMsg(pubLocalSurf, *surfLocal, ros::Time().fromSec(lidarTimestamps[i]), "motor");

		// 找最近点获取原始点
		pcl::KdTreeFLANN<PointType>::Ptr kdtreeLocalMap(new pcl::KdTreeFLANN<PointType>());
		kdtreeLocalMap->setInputCloud(localMap);
		auto &surf = surfs[i];
		for (int j = 0; j < surf->points.size(); ++j) {
			int numNeighbors = 3;
			std::vector<int> pointSearchIdx;
			std::vector<float> pointSearchSqDis;
			PointType pointSel;
			PointType &pointOri = surf->points[j];
			double surfTime = lidarTime2EncodeTime(pointOri.timestamp);
			if (evaluate(traj, surfTime, pos)) {
				double angle = pos.x();
				point2InitEncode(pointOri, pointSel, exTransposeLidar2Encode, exOrientationLidar2Encode, -angle);  // 此处取负角度值，因为码盘按顺时针旋转，在右手系中是负角度方向
				kdtreeLocalMap->nearestKSearch(pointSel, numNeighbors, pointSearchIdx, pointSearchSqDis);

				if (pointSearchSqDis[0] > 0.5) continue;
				
				// 冒泡排序，从小到大排时间
				for (int i = 0; i < 2; ++i) {
					for (int j = 0; j < 2 - i; ++j) {
						if (localMap->points[pointSearchIdx[j]].timestamp > localMap->points[pointSearchIdx[j+1]].timestamp) {
							std::swap(pointSearchIdx[j], pointSearchIdx[j+1]);
						}
					}
				}

				PointType &pointa = localMap->points[pointSearchIdx[0]];
				PointType &pointb = localMap->points[pointSearchIdx[1]];
				PointType &pointc = localMap->points[pointSearchIdx[2]];
				// std::cout << "tb - ta : " << pointb.timestamp - pointa.timestamp << "tc - tb : " << pointc.timestamp - pointb.timestamp << std::endl;
				int aFrameID, aPointID, bFrameID, bPointID, cFrameID, cPointID;
				getFrameAndIndex(pointa.id, aFrameID, aPointID, surfcloudIDs);
				getFrameAndIndex(pointb.id, bFrameID, bPointID, surfcloudIDs);
				getFrameAndIndex(pointc.id, cFrameID, cPointID, surfcloudIDs);

				PointType &pointMatcha = surfs[aFrameID]->points[aPointID];
				PointType &pointMatchb = surfs[bFrameID]->points[bPointID];
				PointType &pointMatchc = surfs[cFrameID]->points[cPointID];
				MatchPair matchPoints;
				matchPoints.pointCur = &pointOri;
				matchPoints.pointa = &pointMatcha;
				matchPoints.pointb = &pointMatchb;
				matchPoints.pointc = &pointMatchc;
				double dist;
				{
					MatchPair &m = matchPoints;
					Eigen::Vector3d p_cur(m.pointCur->x, m.pointCur->y, m.pointCur->z);
					Eigen::Vector3d p_a(m.pointa->x, m.pointa->y, m.pointa->z);
					Eigen::Vector3d p_b(m.pointb->x, m.pointb->y, m.pointb->z);
					Eigen::Vector3d p_c(m.pointc->x, m.pointc->y, m.pointc->z);
					double t_cur = m.pointCur->timestamp;
					double t_a = m.pointa->timestamp;
					double t_b = m.pointb->timestamp;
					double t_c = m.pointc->timestamp;

					evaluate(traj, lidarTime2EncodeTime(t_cur), pos);
					double angle_cur = pos.x();
					evaluate(traj, lidarTime2EncodeTime(t_a), pos);
					double angle_a = pos.x();
					evaluate(traj, lidarTime2EncodeTime(t_b), pos);
					double angle_b = pos.x();
					evaluate(traj, lidarTime2EncodeTime(t_c), pos);
					double angle_c = pos.x();

					Eigen::Matrix<double, 3, 1> trans = exTransposeLidar2Encode;
					Eigen::Quaternion<double> orien(exOrientationLidar2Encode);

					Eigen::Matrix<double, 3, 3> orien2init_cur, orien2init_a, orien2init_b, orien2init_c;
					orien2init_cur << cos(angle_cur), -sin(angle_cur), 0.0,
									sin(angle_cur),  cos(angle_cur), 0.0,
									0.0,        0.0,       1.0;

					orien2init_a << cos(angle_a), -sin(angle_a), 0.0,
									sin(angle_a),  cos(angle_a), 0.0,
									0.0,        0.0,       1.0;

					orien2init_b << cos(angle_b), -sin(angle_b), 0.0,
									sin(angle_b),  cos(angle_b), 0.0,
									0.0,        0.0,       1.0;
					
					orien2init_c << cos(angle_c), -sin(angle_c), 0.0,
									sin(angle_c),  cos(angle_c), 0.0,
									0.0,        0.0,       1.0;

					p_cur = orien * p_cur + trans;
					p_cur = orien2init_cur * p_cur;

					p_a = orien * p_a + trans;
					p_a = orien2init_a * p_a;

					p_b = orien * p_b + trans;
					p_b = orien2init_b * p_b;

					p_c = orien * p_c + trans;
					p_c = orien2init_c * p_c;

					p_b = p_a - p_b;
					p_c = p_a - p_c;
					p_cur = p_cur - p_a;
					p_b = p_b.cross(p_c);
					p_b.normalize();

					dist = p_b.dot(p_cur);
					// std::cout << "dist : " << dist << std::endl;
				}

				if (dist < 0.5) {
					matchPairs.push_back(matchPoints);
				}

			}	
		}
		if (i % 10 == 0) {
			std::cout << "ten frames find time : " << tictoc.Toc() << " ms\n";
		}
		// if (count == 30) break;
	}
	std::cout << "find correspond time : " << tictoc.Toc() << " ms\n";
}

void optimizeWithCloud(std::shared_ptr<kontiki::trajectories::SplitTrajectory> &traj) {
	TicToc tictoc;
	tictoc.Tic();
	std::shared_ptr<kontiki::sensors::VLP16LiDAR> lidar_;
	lidar_ = std::make_shared<kontiki::sensors::VLP16LiDAR>();
	lidar_->set_relative_orientation(Eigen::Quaterniond(exOrientationLidar2Encode));
	lidar_->set_relative_position(exTransposeLidar2Encode);
	lidar_->LockRelativeOrientation(true);
	lidar_->LockRelativePosition(true);
	lidar_->LockTimeOffset(false);
	lidar_->set_max_time_offset(timeDelayMax);
	lidar_->set_time_offset(timeDelay);

	// 固定控制点，只优化时延
	traj->Lock(true);
	// 添加点-点残差
	std::shared_ptr<R3TrajEstimator> estimator_split;  // 用于优化
	estimator_split = std::make_shared<R3TrajEstimator>(traj->R3Spline());
	std::vector< std::shared_ptr<Point2PlaneMeasurement>>  msg_list_; // 保存测量值，需要保存智能指针
	int matchSize = matchPairs.size();
	for (int i = 0; i < matchSize; i += 1) {
		MatchPair &m = matchPairs[i];
		const double weight = 1;
		Eigen::Vector3d p_cur(m.pointCur->x, m.pointCur->y, m.pointCur->z);
		Eigen::Vector3d p_a(m.pointa->x, m.pointa->y, m.pointa->z);
		Eigen::Vector3d p_b(m.pointb->x, m.pointb->y, m.pointb->z);
		Eigen::Vector3d p_c(m.pointc->x, m.pointc->y, m.pointc->z);
		double t_cur = m.pointCur->timestamp;
		double t_a = m.pointa->timestamp;
		double t_b = m.pointb->timestamp;
		double t_c = m.pointc->timestamp;
		auto msp = std::make_shared<Point2PlaneMeasurement> (lidar_, p_cur, p_a, p_b, p_c, t_cur, t_a, t_b, t_c, 1.0, weight);
		msg_list_.push_back(msp);
		estimator_split->template AddMeasurement<Point2PlaneMeasurement>(msp);
	}
	std::cout << "add lidar measurements time : " <<  tictoc.Toc() << " ms\n";
	// 求解
	ceres::Solver::Summary summary = estimator_split->Solve(50, false);
	std::cout << "solve with lidar \n";
	std::cout << summary.BriefReport() << std::endl;
	std::cout << "\n\n ------------------------------\n";
	std::cout << summary.FullReport() << std::endl;
	std::cout << "ceres sovle time : " <<  tictoc.Toc() << " ms\n";

	timeDelay = lidar_->time_offset();
	exOrientationLidar2Encode = lidar_->relative_orientation().toRotationMatrix();
	exTransposeLidar2Encode = lidar_->relative_position();
	std::cout << "\n time delay : " << timeDelay * 1000 << " ms" << std::endl;
	std::cout << "\n postion : " << lidar_->relative_position() << std::endl;
	std::cout << "\n orientation : " << lidar_->relative_orientation().toRotationMatrix().eulerAngles(2, 1, 0).transpose() << std::endl;
}

template<typename T>
void readParam(ros::NodeHandle &nh, std::string name, T &param, T def) {
	if (nh.getParam(name, param)) {
		ROS_INFO_STREAM("Loaded " << name << " : " << param);
	} else {
		param = def;
		ROS_INFO_STREAM("Loaded default " << name << " : " << param);
	}
}

// end namespace leCalib
int main(int argc, char** argv) {
	ros::init(argc, argv, "le_calib");
	ros::NodeHandle nh("~");
	ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_full_after_calibration", 2);
	ros::Publisher pubLaserCloudCorner = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_corner_after_calibration", 2);
	ros::Publisher pubLaserCloudSurf = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_surf_after_calibration", 2);
	ros::Publisher pubLaserCloudOrign = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_orign", 2);
	ros::Publisher &pubLocalMapCopy = pubLocalMap;
	ros::Publisher &pubLocalSurfCopy = pubLocalSurf;
	pubLocalMapCopy = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_local_map", 2);
	pubLocalSurfCopy = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_local_surf", 2);

	std::vector<double> para_rpxy(4); // 外参
	// 读取数据包数据
	std::string path, encode_topic, lidar_topic;
	double bag_start = 0.0;
	double bag_durr = 30;
	double knot_distance = 0.05;
	double &time_delay_max = timeDelayMax;
	double &time_delay = timeDelay;
	double &angle_dist1 = angleDist1;
	double &angle_dist2 = angleDist2;

	// 加载参数
	para_rpxy = nh.param("extrinsic", std::vector<double>(4, 0.0));
	ROS_INFO_STREAM("Loaded roll : " << para_rpxy[0]);
	ROS_INFO_STREAM("Loaded pitch : " << para_rpxy[1]);
	ROS_INFO_STREAM("Loaded x : " << para_rpxy[2]);
	ROS_INFO_STREAM("Loaded y : " << para_rpxy[3]);

	readParam(nh, "time_delay_max", time_delay_max, 0.5);
	readParam(nh, "time_delay", time_delay, 0.0);
	readParam(nh, "path", path, std::string("/home/dut-zxw/zxw/data_bag/lidar_encode/xuanzhuanbiaoding/chang_speed_max_180_per_s.bag"));
	readParam(nh, "encode_topic", encode_topic, std::string("/INNFOS/actuator_states"));
	readParam(nh, "lidar_topic", lidar_topic, std::string("/velodyne_points"));
	readParam(nh, "bag_start", bag_start, 0.0);
	readParam(nh, "bag_durr", bag_durr, -1.0);
	readParam(nh, "knot_distance", knot_distance, 0.05);
	readParam(nh, "angle_dist1", angle_dist1, 170.0);
	readParam(nh, "angle_dist2", angle_dist2, 190.0);
	angle_dist1 = deg2rad(angle_dist1);
	angle_dist2 = deg2rad(angle_dist2);

	para_rpxy[0] = deg2rad(para_rpxy[0]);
	para_rpxy[1] = deg2rad(para_rpxy[1]);

	Eigen::Vector3d &exTranspose = exTransposeLidar2Encode;
	exTranspose = Eigen::Vector3d(para_rpxy[2], para_rpxy[3], 0);
	Eigen::Matrix3d &exOrientation = exOrientationLidar2Encode;
	exOrientation = pr2R(para_rpxy[1], para_rpxy[0]);

	readDataBag(path, encode_topic, lidar_topic, bag_start, bag_durr);
	adjustDataset();

	// 计算点云中每个点的时间戳，提取出特征点
	std::vector<pcl::shared_ptr<PointCloud>> cornerClouds, surfClouds;

	for (int i = 0; i < lidarData.size(); ++i) {
		cornerClouds.push_back(pcl::make_shared<PointCloud>());
		surfClouds.push_back(pcl::make_shared<PointCloud>());
		extracFeatureAndCalculateTime(lidarData[i], lidarTimestamps[i], cornerClouds.back(), surfClouds.back());
		// std::cout << "frame " << i << std::endl;
		// std::cout << "full   Cloud size : " << lidarData[i]->points.size() << std::endl;
		// std::cout << "corner Cloud size : " << cornerClouds.back()->points.size() << std::endl;
		// std::cout << "surf Cloud size : " << surfClouds.back()->points.size() << std::endl << std::endl;
	}

	// 根据码盘数据拟合出码盘的数据
	std::shared_ptr<kontiki::trajectories::SplitTrajectory> traj;
	initTrajectory(traj, knot_distance);
	optimizeEncodeAngles(traj);

	// 输出样条曲线上的值
    std::vector<std::pair<double, double>> spline_value;
    for (double i = encodeData[0].timestamp; i < encodeData.back().timestamp; i += 0.01) {
        double time = i;
        Eigen::Vector3d p;
        if (evaluate(traj, time, p)) {
            spline_value.push_back({time - encodeData[0].timestamp, p.x()});
        } else {
            std::cout << "time error\n";
        }
    }
    std::cout << "write file, encodeData.size = " << encodeData.size() << "  spline_value.size = " << spline_value.size() << std::endl;;
    std::ofstream outfile;
    // 测量数据
    outfile.open("/home/dut-zxw/zxw/le_calib_ws/measure.txt");
    for (auto v : encodeData) {
        outfile << v.timestamp - encodeData[0].timestamp << " " << rad2deg(v.angle) << std::endl;
    }
    outfile.close();
    // 样条估计数据
    outfile.open("/home/dut-zxw/zxw/le_calib_ws/spline.txt");

    for (auto v : spline_value) {
        outfile << v.first << " " << rad2deg(v.second) << std::endl;
    }
	
	// 计算点的id，为找匹配对作准备
	calculateCloudIDs(surfClouds, surfcloudIDs);
	
	int iterCount = 50;
	for (int i = 0; i < iterCount; ++i) {
		// 找到所有匹配对
		matchPairs.clear();
		std::cout << "start find corressponds\n";
		findCorressponds(surfClouds, traj, 0, surfClouds.size() - 1, 0);
		std::cout << "match pair size : " << matchPairs.size() << std::endl;

		// 使用匹配对优化时延
		optimizeWithCloud(traj);
	}

	// 发布点云
	ros::Rate r(10);
	for (int i = 0; i < lidarData.size(); ++i) {
		pcl::shared_ptr<PointCloudIntensity> fullCloud = pcl::make_shared<PointCloudIntensity>();
		pcl::shared_ptr<PointCloudIntensity> cornerCloud = pcl::make_shared<PointCloudIntensity>();
		pcl::shared_ptr<PointCloudIntensity> surfCloud = pcl::make_shared<PointCloudIntensity>();
		auto &orignFull = lidarData[i];
		auto &orignCorner = cornerClouds[i];
		auto &orignSurf = surfClouds[i];
		transformCloud2Init(*orignFull, *fullCloud, exTransposeLidar2Encode, exOrientationLidar2Encode, traj);
		transformCloud2Init(*orignCorner, *cornerCloud, exTransposeLidar2Encode, exOrientationLidar2Encode, traj);
		transformCloud2Init(*orignSurf, *surfCloud, exTransposeLidar2Encode, exOrientationLidar2Encode, traj);
		ros::Time stamp = ros::Time().fromSec(lidarTimestamps[i]);
		PublishCloudMsg(pubLaserCloudFull, *fullCloud, stamp, "motor");
		PublishCloudMsg(pubLaserCloudCorner, *cornerCloud, stamp, "motor");
		PublishCloudMsg(pubLaserCloudSurf, *surfCloud, stamp, "motor");
		// PublishCloudMsg(pubLaserCloudOrign, *orignFull, stamp, "motor");

		r.sleep();
	}
	
	std::cout << "\n\n\ncalib ok\n";
	return 0;
}
