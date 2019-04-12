//2019-3-18
//这个程序能够对图像进行二值化 并且找到二值化图像的质心
//4-12 已经实现能够对最大的n个轮廓提取质心
//4-12 需要对视频能够实现提取质心 可以实现
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <algorithm>
using namespace std;
using namespace cv;

#define CONTOUR_SIZE 5
#define Filter_W 200
#define Filter_H 200
#define Video_W 768
#define Video_H 576

# if 0
int Otsu(Mat &image)
{
	int width = image.cols;
	int height = image.rows;
	int x = 0, y = 0;
	int pixelCount[256];
	float pixelPro[256];
	int i, j, pixelSum = width * height, threshold;

	uchar *data = (uchar*)image.data;

	for (i = 0; i < 256; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	for (i = y; i < height; i++)
	{
		for (j = x; j < width; j++)
		{
			pixelCount[data[i * image.step + j]]++;
		}
	}

	for (i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
	}

	float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
	for (i = 0; i < 256; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for (j = 0; j < 256; j++)
		{
			if (j <= i)
			{
				w0 += pixelCount[j];
				u0tmp += j * pixelPro[j];
			}
			else
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}

		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		u = u0tmp + u1tmp;

		deltaTmp = w0 * (u0 - u) * (u0 - u) + w1 * (u1 - u) * (u1 - u);

		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	return threshold;
}

int main()
{
	Mat White = imread("gmm.png");
	int threshold_white = 120;//Otsu(White);
	cout << "最佳阈值" << threshold_white << endl;
	Mat thresholded = Mat::zeros(White.size(), White.type());//返回指定大小和类型的零数组
	threshold(White, thresholded, threshold_white, 255, CV_THRESH_BINARY);
	imshow("二值化", thresholded);
	waitKey(0);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;//4维向量
	findContours(thresholded, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);//找到轮廓

	int count = 0;
	Point pt[10];
	Moments moment;//矩
	vector<Point> Center;
	for (int i = 0; i >= 0; i = hierarchy[i][0])//这是什么意思？？？
	{
		Mat temp(contours.at(i));
		Scalar color(0, 0, 255);
		moment = moments(temp, false);
		if (moment.m00 != 0)
		{
			pt[i].x = cvRound(moment.m10 / moment.m00);
			pt[i].y = cvRound(moment.m01 / moment.m00);
		}
		Point p = Point(pt[i].x, pt[i].y);
		circle(White, p, 1, color, 1, 8);
		count++;
		Center.push_back(p);
	}
	cout << "重心点个数:" << Center.size() << endl;
	cout << "轮廓数量" << contours.size() << endl;
	//imwrite("Center.tif", White);
}
#endif

//这个主函数能够实现对图片的操作
#if 0
int main()
{
	Mat img, imgGray, result;
	img = imread("test.jpg");

	if (!img.data)
	{
		cout << "Please input image path" << endl;
		return 0;
	}
	imshow("原图", img);
	cvtColor(img, imgGray, CV_BGR2GRAY);
	imshow("灰度图", imgGray);
	threshold(imgGray, result, 100, 255, CV_THRESH_BINARY);
	imshow("二值化", result);

#if 0
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;//4维向量
	findContours(result, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);//找到轮廓

	int count = 0;
	Point pt[10];
	Moments moment;//矩
	vector<Point> Center;
	for (int i = 0; i >= 0; i = hierarchy[i][0])//这是什么意思？？？
	{
		Mat temp(contours.at(i));
		Scalar color(0, 0, 0);
		moment = moments(temp, false);
		if (moment.m00 != 0)
		{
			pt[i].x = cvRound(moment.m10 / moment.m00);
			pt[i].y = cvRound(moment.m01 / moment.m00);

			Point p = Point(pt[i].x, pt[i].y);
			circle(img, p, 10, color, 1, 8);
			count++;
			Center.push_back(p);
		}
	}
	cout << "重心点个数:" << Center.size() << endl;
	for (int i = 0; i < Center.size(); i++)
	{
		cout << Center[i].x << " "  << Center[i].y << endl;
		rectangle(img, Rect(Center[i].x, Center[i].y, 100, 100), Scalar(255, 255, 255), 2);
	}
	cout << "轮廓数量" << contours.size() << endl;
#endif

	Mat canny_output;
	vector<vector<Point>> contours;

	vector<Vec4i> hierarchy;//4维向量
	//利用canny算法检测边缘
	Canny(result, canny_output, 30, 90, 3);//这个参数的设置?
	imshow("canny", canny_output);
	//查找轮廓
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
	//要是可以对contours里的size进行排序就好了
	//sort(contours.begin(), contours.end());默认的sort应该是只能对int类型进行排序
	vector<int> temp_contour;
	for (int i = 0; i < contours.size(); i++)
	{
		temp_contour.push_back(contours[i].size());
	}
	sort(temp_contour.rbegin(), temp_contour.rend());//从大到小排序
	vector<int> res_contour;
	for (int i = 0; i < CONTOUR_SIZE; i++)
	{
		res_contour.push_back(temp_contour[i]);
	}
	vector<vector<Point>> best_contours;//最好的轮廓
	for (int i = 0; i < res_contour.size(); i++)
	{
		for (auto it = contours.begin(); it != contours.end();)
		{
			if (res_contour[i] == it->size())
			{
				best_contours.push_back(*it);
				it = contours.erase(it);
				break;
			}
			else
			{
				it++;
			}
		}
	}

	/*
	int max = 0, max_index = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (max < contours[i].size())
		{
			max = contours[i].size();
			max_index = i;
		}
	}
	*/
	//计算轮廓矩
	vector<Moments> mu(best_contours.size());
	for (int i = 0; i < best_contours.size(); i++)
	{
		mu[i] = moments(best_contours[i], false);
	}
	//计算轮廓的质心
	vector<Point2f> mc(best_contours.size());
	for (int i = 0; i < best_contours.size(); i++)
	{
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}
	//画轮廓及其质心并显示
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
#if 1
	for (int i = 0; i < best_contours.size(); i++)
	{
		Scalar color = Scalar(255, 0, 0);
		drawContours(drawing, best_contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i], 5, Scalar(0, 0, 255), -1, 8, 0);
		rectangle(drawing, boundingRect(best_contours.at(i)), cvScalar(0, 255, 0));
		cout << "x:" << mc[i].x << "y:" << mc[i].y << endl;
		char tam[100];
		sprintf_s(tam, "(%0.0f, %0.0f)", mc[i].x, mc[i].y);
		putText(drawing, tam, Point(mc[i].x, mc[i].y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, cvScalar(0, 255, 0), 1);
	}
#else

	Scalar color = Scalar(255, 0, 0);
	drawContours(drawing, contours, max_index, color, 2, 8, hierarchy, 0, Point());
	circle(drawing, mc[max_index], 5, Scalar(0, 0, 255), -1, 8, 0);
	rectangle(drawing, boundingRect(contours.at(max_index)), cvScalar(0, 255, 0));
	cout << "x:" << mc[max_index].x << "y:" << mc[max_index].y << endl;
	char tam[100];
	sprintf_s(tam, "(%0.0f, %0.0f)", mc[max_index].x, mc[max_index].y);
	putText(drawing, tam, Point(mc[max_index].x, mc[max_index].y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, cvScalar(0, 255, 0), 1);//终于能搞到最想要的东西了
#endif

	imshow("Contours", drawing);
	waitKey(0);
	return 0;
}

#else 

//这个主函数能够实现对视频的操作


int main()
{
	VideoCapture video;
	video.open("vtest.avi");
	if (!video.isOpened())
	{
		printf("No Video\n");
		getchar();
		return -1;
	}

	Mat frame;
	int frame_num = video.get(CAP_PROP_FRAME_COUNT);
	cout << "total frame number is: " << frame_num << endl;
	//int width = video.get(CAP_PROP_FRAME_WIDTH);
	//int height = video.get(CAP_PROP_FRAME_HEIGHT);
	//GMM初始化设置
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));//?
	Mat bsmMOG2;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();//?

	while (video.read(frame))
	{
		resize(frame, frame, Size(Video_W, Video_H));
		//进行高斯建模
		pMOG2->apply(frame, bsmMOG2);
		morphologyEx(bsmMOG2, bsmMOG2, MORPH_OPEN, kernel);//?
		resize(bsmMOG2, bsmMOG2, Size(Video_W, Video_H));
		imshow("GMM", bsmMOG2);

		Mat canny_output;
		vector<vector<Point>> contours;

		vector<Vec4i> hierarchy;//4维向量
		//利用canny算法检测边缘
		Canny(bsmMOG2, canny_output, 30, 90, 3);//这个参数的设置?
		imshow("canny", canny_output);
		//查找轮廓
		findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		
		//要是可以对contours里的size进行排序就好了
		//sort(contours.begin(), contours.end());默认的sort应该是只能对int类型进行排序
		//找到最好的几个轮廓
		vector<int> temp_contour;
		for (int i = 0; i < contours.size(); i++)
		{
			temp_contour.push_back(contours[i].size());
		}
		sort(temp_contour.rbegin(), temp_contour.rend());//从大到小排序
		vector<int> res_contour;
		for (int i = 0; i < CONTOUR_SIZE; i++)
		{
			res_contour.push_back(temp_contour[i]);
		}
		vector<vector<Point>> best_contours;//最好的轮廓
		for (int i = 0; i < res_contour.size(); i++)
		{
			for (auto it = contours.begin(); it != contours.end();)
			{
				if (res_contour[i] == it->size())
				{
					best_contours.push_back(*it);
					it = contours.erase(it);
					break;
				}
				else
				{
					it++;
				}
			}
		}

		//计算轮廓矩
		vector<Moments> mu(best_contours.size());
		for (int i = 0; i < best_contours.size(); i++)
		{
			mu[i] = moments(best_contours[i], false);
		}
		//计算轮廓的质心
		vector<Point2f> mc(best_contours.size());
		for (int i = 0; i < best_contours.size(); i++)
		{
			mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}
		//画轮廓及其质心并显示
		Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

		for (int i = 0; i < best_contours.size(); i++)
		{
			Scalar color = Scalar(255, 0, 0);
			drawContours(drawing, best_contours, i, color, 2, 8, hierarchy, 0, Point());
			circle(drawing, mc[i], 5, Scalar(0, 0, 255), -1, 8, 0);
			rectangle(drawing, boundingRect(best_contours.at(i)), cvScalar(0, 255, 0));
			cout << "x:" << mc[i].x << "y:" << mc[i].y << endl;
			char tam[100];
			sprintf_s(tam, "(%0.0f, %0.0f)", mc[i].x, mc[i].y);
			putText(drawing, tam, Point(mc[i].x, mc[i].y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, cvScalar(0, 255, 0), 1);
		}
		
		imshow("Contours", drawing);
		if (waitKey(10) == 'q')
			break;
	}
	return 0;
}
#endif