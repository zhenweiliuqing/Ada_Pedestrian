//2019-3-15
//�ܹ�����Adaboost��xml�ļ� ����ʶ��Ŀ�� �����󱨺ܶ� ��Ҫ�ø�˹������ģ�˳�

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

# if 1

//adaboost �� ��Ƶ��� �� ��˹��������

int main()
{
	//����ѵ��ģ��
	CascadeClassifier cascade;
	cascade.load("cascade.xml");

	//��ʼ����Ƶ����
	VideoCapture video;
	video.open("playground.avi");
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
	//GMM��ʼ������
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));//?
	Mat bsmMOG2;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();//?

	//��������
	VideoWriter out;
	out.open("test_result.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, Size(512, 288), true);//�����������

	//������
	while (video.read(frame))
	{
		resize(frame, frame, Size(256, 144));
		//���и�˹��ģ
		pMOG2->apply(frame, bsmMOG2);
		morphologyEx(bsmMOG2, bsmMOG2, MORPH_OPEN, kernel);//?
		resize(bsmMOG2, bsmMOG2, Size(256, 144));
		imshow("MOG2", bsmMOG2);
		
		//��ȡ��Ե
		Mat canny_output;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;//4ά����
		//����canny�㷨����Ե
		Canny(bsmMOG2, canny_output, 30, 90, 3);//�������������?
		imshow("canny", canny_output);
		//��������
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
			//����������
			Moments mu;
			mu = moments(contours[max_index], false);
			//��������������
			Point2f mc;
			mc = Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);
			//�������������Ĳ���ʾ
			Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

			Scalar color = Scalar(255, 0, 0);
			drawContours(drawing, contours, max_index, color, 2, 8, hierarchy, 0, Point());
			circle(drawing, mc, 5, Scalar(0, 0, 255), -1, 8, 0);
			rectangle(drawing, boundingRect(contours.at(max_index)), cvScalar(0, 255, 0));
			cout << "x:" << mc.x << "y:" << mc.y << endl;
			char tam[100];
			sprintf_s(tam, "(%0.0f, %0.0f)", mc.x, mc.y);
			putText(drawing, tam, Point(mc.x, mc.y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, cvScalar(0, 255, 0), 1);//�����ܸ㵽����Ҫ�Ķ�����
			x = mc.x;
			y = mc.y;
			imshow("Contours", drawing);
		}

		vector<Rect> found, found_1, found_filtered;
		//���м��
		cascade.detectMultiScale(frame, found, 1.1, 3, 0);//�������������
		//��һ��ɸѡ
		for (int i = 0; i < found.size(); i++)
		{
			if (found[i].x > 0 && found[i].y > 0 && (found[i].x + found[i].width) < frame.cols\
				&& (found[i].y + found[i].height) < frame.rows)
				found_1.push_back(found[i]);
		}
		//�ڶ���ɸѡ ȥǶ��
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
		//�����ο� ����΢��
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
			if (x > r.tl().x && y > r.tl().y && x < r.br().x && y < r.br().y)
				rectangle(frame, r.tl(), r.br(), Scalar(0, 0, 255), 2);
		}

		imshow("detect result", frame);
		out << frame;

		if (waitKey(1) == 'q')
			break;
	}
	video.release();
	out.release();
	return 0;
}

#endif
