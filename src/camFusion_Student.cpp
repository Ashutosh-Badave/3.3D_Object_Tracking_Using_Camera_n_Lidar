
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> KeyPoints_BB;
    float dist=0,mean,variance = 0.0,stdDeviation;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        if (boundingBox.roi.contains(kptsCurr.at(it->trainIdx).pt))
        {       
            KeyPoints_BB.push_back(*it);
        }
    }

    for (auto it1 = KeyPoints_BB.begin(); it1 != KeyPoints_BB.end(); ++it1)
    { 
        cv::KeyPoint kpCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it1->queryIdx);
        dist += cv::norm(kpCurr.pt - kpPrev.pt);
    }
    mean = dist/KeyPoints_BB.size();
    cout<< "mean: "<<mean<<endl;
    dist = 0;
    for (auto it2 = KeyPoints_BB.begin(); it2 != KeyPoints_BB.end(); ++it2)
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it2->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it2->queryIdx);
        dist = cv::norm(kpCurr.pt - kpPrev.pt);
        //variance +=pow(dist-mean,2);
        if (dist< mean*2)
            boundingBox.kptMatches.push_back(*it2);
    }
    
    cout<<"Matched Keypoints size with outliers: "<<KeyPoints_BB.size()<<endl;
    cout<<"Matched Keypoints size mean based outliers removal: "<<boundingBox.kptMatches.size()<<endl;
  
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    double medianDistRatio;
    std::sort(distRatios.begin(), distRatios.end());

    int size = distRatios.size();

    if (size % 2 == 0)
        medianDistRatio = (distRatios[size/2 -1] + distRatios[size/2])/2;

    medianDistRatio = distRatios[size/2];

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
    cout << "Time to collision by camera = " << TTC << "s" << endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1/frameRate;
    double laneWidth = 4.0;
    double minXPrev = 1e9, minXCurr = 1e9;
    //float sum = 0.0, mean, variance = 0.0, stdDeviation;
    std::vector<double> PRev_Xpoints,Curr_XPoints;

    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it)
    {
        PRev_Xpoints.push_back(it->x);
        //cout << "Prev x = " << it->x << "s" << endl;
        //sum += it->x;
        //minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }
    //mean = sum/PRev_Xpoints.size();

    sort(PRev_Xpoints.begin(), PRev_Xpoints.end());
    /*cout<<"Prev Points"<<endl;
    for (int i=0;i<5;i++)
        cout<< PRev_Xpoints[i] <<endl;*/
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        Curr_XPoints.push_back(it->x);
        //cout << "Current x = " << it->x << "s" << endl;
        //minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }
    sort(Curr_XPoints.begin(), Curr_XPoints.end());

    int PrevMedian_Ind = static_cast<int>(PRev_Xpoints.size()/2);
    int CurrMedian_Ind = static_cast<int>(Curr_XPoints.size()/2);

    TTC = Curr_XPoints[CurrMedian_Ind] * dT / (PRev_Xpoints[PrevMedian_Ind] - Curr_XPoints[CurrMedian_Ind]);
    cout << "Time to collision by Lidar = " << TTC << "s" << endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  cv::KeyPoint curr_keypoint, prev_keypoint;
    const int PrevBBSize = prevFrame.boundingBoxes.size();
    const int CurrBBSize = currFrame.boundingBoxes.size();
    cv::Mat count = cv::Mat::zeros(PrevBBSize, CurrBBSize, CV_32S);
    //std::multimap<int, int> Allmatches;
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        //cout << "matches between: " << it->queryIdx << ", " << it->trainIdx << endl;
        curr_keypoint = currFrame.keypoints[it->trainIdx];
        prev_keypoint = prevFrame.keypoints[it->queryIdx];

        for (int i = 0; i < PrevBBSize; i++)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(prev_keypoint.pt))
            {

                for (int j = 0; j < CurrBBSize; j++)
                {
                    if (currFrame.boundingBoxes[j].roi.contains(curr_keypoint.pt))
                    {
                        count.at<int>(i,j)=count.at<int>(i,j)+1;
                        //cout << "pair: "<<i<<", "<<j << endl;
                        //cout<<"count: "<<count.at<int>(i,j)<<endl;

                    }

                }
            }
        }
    }
    for (int i = 0; i < PrevBBSize;i++)
    {
        int match_id = 0;
        int maxcount = 0;
        for (int j = 0; j < CurrBBSize;j++)
        {

            if(count.at<int>(i,j)> maxcount)
            {
                maxcount = count.at<int>(i,j);
                match_id = j;
            }
        }
        //cout<<"##After maxcount"<<endl;
        //cout << "pair: "<<i<<", "<<match_id << endl;
        //cout<<"count: "<<count.at<int>(i,match_id)<<endl;
        bbBestMatches[i] = match_id;
    }
    /*for(int i = 0; i < prevFrame.boundingBoxes.size();i++)
    {
        cout << " BoundingBox : " << i << " from 1st frame matches with BoundingBox : " <<
    bbBestMatches[i] << " from frame 2 " << endl;
    }*/
}
