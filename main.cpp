//2019-3-30
//能够加载Adaboost的xml文件 并且识别到目标 
//用了新的模型 误报只减少了一点点 加了框的大小以减少误报 有一点用
//总体感觉还是背景有很多误报 通过背景滤除和框的大小的选择 减少了很多 
//识别的准确率还是不够高 训练的样本很好 但是实际中 人姿态变化很大 而且 从大到小 变化很多 只能在比较符合训练集大小的情况下识别出来

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/objdetect.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

const string img_save_path = "ground_result/";
# if 1

//adaboost 加 视频检测 加 高斯背景过滤

int main()
{
	//载入训练模型
	CascadeClassifier cascade;
	cascade.load("cascade_329.xml");

	//初始化视频输入
	VideoCapture video;
	video.open("fire.mp4");
	if (!video.isOpened())
	{
		printf("No Video\n");
		getchar();
		return -1;
	}

	Mat frame;
	int frame_num = video.get(CAP_PROP_FRAME_COUNT);
	cout << "total frame number is: " << frame_num << endl;
	int width = video.get(CAP_PROP_FRAME_WIDTH);
	int height = video.get(CAP_PROP_FRAME_HEIGHT);
	//GMM初始化设置
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));//?
	Mat bsmMOG2;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();//?

	//保存检测结果
	VideoWriter out;
	out.open("res_fire.mp4", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, Size(320, 180), true);//这个参数？？
	
	//计数截取的框框
	int count = 0;
	char saveName[256];
	//检测过程
	while (video.read(frame))
	{
		resize(frame, frame, Size(320, 180));
		//进行高斯建模
		pMOG2->apply(frame, bsmMOG2);
		morphologyEx(bsmMOG2, bsmMOG2, MORPH_OPEN, kernel);//?
		resize(bsmMOG2, bsmMOG2, Size(320, 180));
		imshow("MOG2", bsmMOG2);
		
		//提取边缘
		Mat canny_output;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;//4维向量
		//利用canny算法检测边缘
		Canny(bsmMOG2, canny_output, 30, 90, 3);//这个参数的设置?
		imshow("canny", canny_output);
		//查找轮廓
		findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		int x = 0, y = 0;
		if (contours.size() != 0)
		{
			int max = 0, max_index = 0;
			for (int i = 0; i < contours.size(); i++)
			{
				if (max < contours[i].size())
				{
					max = contours[i].size();
					max_index = i;
				}
			}
			//计算轮廓矩
			Moments mu;
			mu = moments(contours[max_index], false);
			//计算轮廓的质心
			Point2f mc;
			mc = Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);
			//画轮廓及其质心并显示
			Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

			Scalar color = Scalar(255, 0, 0);
			drawContours(drawing, contours, max_index, color, 2, 8, hierarchy, 0, Point());
			circle(drawing, mc, 5, Scalar(0, 0, 255), -1, 8, 0);
			rectangle(drawing, boundingRect(contours.at(max_index)), cvScalar(0, 255, 0));
			cout << "x:" << mc.x << "y:" << mc.y << endl;
			char tam[100];
			sprintf_s(tam, "(%0.0f, %0.0f)", mc.x, mc.y);
			putText(drawing, tam, Point(mc.x, mc.y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, cvScalar(0, 255, 0), 1);//终于能搞到最想要的东西了
			x = mc.x;
			y = mc.y;
			imshow("Contours", drawing);
		}

		vector<Rect> found, found_1, found_filtered;
		//进行检测
		cascade.detectMultiScale(frame, found, 1.1, 3, 0);//这个参数怎设置
		//第一波筛选
		for (int i = 0; i < found.size(); i++)
		{
			if (found[i].x > 0 && found[i].y > 0 && (found[i].x + found[i].width) < frame.cols\
				&& (found[i].y + found[i].height) < frame.rows)
				found_1.push_back(found[i]);
		}
		//第二波筛选 去嵌套
		for (int i = 0; i < found_1.size(); i++)
		{
			Rect r = found_1[i];
			int j = 0;
			for (; j < found_1.size(); j++)
			{
				if (j != i && (r & found_1[j]) == found_1[j])
					break;
			}
			if (j == found_1.size())
				found_filtered.push_back(r);
		}
		//画矩形框 进行微调
		for (int i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width * 0.1);
			r.width = cvRound(r.width * 0.8);
			r.y += cvRound(r.height * 0.1);
			r.height = cvRound(r.height * 0.8);
		}

		for (int i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
#if 1 //进行滤除
			int lx = r.tl().x, ly = r.tl().y, rx = r.br().x, ry = r.br().y;
			if (x > lx && y > ly && x < rx && y < ry && r.width < 100 && r.height < 100)
			{
				rectangle(frame, r.tl(), r.br(), Scalar(0, 0, 255), 2);
				//把框出来的目标输出到一个文件夹 分析误检测的特征
				//Mat imgROI = frame(Rect(lx, ly, rx - lx, ry - ly));
				//Mat imgROI = frame(r);
				//sprintf_s(saveName, "cut_%04d.jpg", ++count);
				//string img_path = img_save_path + string(saveName);
				//imwrite(img_path, imgROI);
				out << frame;
			}
#else //不进行滤除
			rectangle(frame, r.tl(), r.br(), Scalar(0, 0, 255), 2);
			out << frame;
#endif
		}

		imshow("detect result", frame);
		

		if (waitKey(1) == 'q')
			break;
	}
	video.release();
	out.release();
	return 0;
}

#endif
