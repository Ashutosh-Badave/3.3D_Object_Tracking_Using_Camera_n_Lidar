# Final Report 

## FP.1 Match 3D Objects
> 1. Looped over every matched keypoint, current and previous keypoints are copied in local variables.
> 2. Inside first loop , second loop is started on previous frame boudingboxes.
> 3. Inside second loop, codition of Boundingbox region of intrest contains keypoint is checked, If keypoint is inside then third loop on current Boundingbox will start.
>4. In third loop , same condition of ROI is checked and if its true then count is increamented for respective matched pair.
>5. Simillary all keypoints are checked and all bounding box pair is found.

> Code for the same:    
    
    
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

---

## FP. 2 Compute Lidar-based TTC
1. All the x coordinates of previous frame lidar point and current frame lidar points are taken into another vectors.
2. Sorted those vectors.
3. Took their median to avoid outliers consideration in TTC calculation.

{

    
    double dT = 1/frameRate;
    double laneWidth = 4.0;
    double minXPrev = 1e9, minXCurr = 1e9;
    //float sum = 0.0, mean, variance = 0.0, stdDeviation;
    std::vector<double> PRev_Xpoints,Curr_XPoints;

    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it)
    {
        PRev_Xpoints.push_back(it->x);
    }

    sort(PRev_Xpoints.begin(), PRev_Xpoints.end());
    
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        Curr_XPoints.push_back(it->x);
    }
    sort(Curr_XPoints.begin(), Curr_XPoints.end());

    int PrevMedian_Ind = static_cast<int>(PRev_Xpoints.size()/2);
    int CurrMedian_Ind = static_cast<int>(Curr_XPoints.size()/2);

    TTC = Curr_XPoints[CurrMedian_Ind] * dT / (PRev_Xpoints[PrevMedian_Ind] - Curr_XPoints[CurrMedian_Ind]);
    cout << "Time to collision by Lidar = " << TTC << "s" << endl;
}

---
## FP.3 Associate KEypoint COrrespodences with Bounding Boxes
> To make TTC calculations robust , removing the outlier with below approch.
1. A vector created to hold matched keypoints which are inside the current bounding box.
2. A loop is run to calculate the distance between each of current keypoint and previous keypoint, from that mean of distance is calculated.
3. If current keypoint is away from the threshold value then it will not be considered, others will be added as boundingbox match keypoints.

> Code for same:

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

---
## FP.4 Compute Camera-based TTC

> Create vector to store distance ratios for all keypoints between curr. and prev. frame.

> Loop over keypoint matches , check the distance ratio and calculate the TTC with the median to make it robust. 

    vector<double> distRatios;  
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
        } 
    }     

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

---
## FP.5 Performance Evaluation 1
> In my opinion, TTC estimation for frame 4,5 and 11 does not seems plausible.  
> For frame 4, TTC with Lidar is 14.091. for frame 5, it is 16.684 and for frame 11 it is 12.8086. 
> I have added Lidar Top view and 3D object pictures for respective frames.
> If we manually calculate the TTC with lidar formula, it might gives different results.

> This is happening because we are not sure about exact position of nearest point on front vehicle. This can be avoided if we use Radar sensor as it provides velocity based on doppler effect.
---
## FP.6 PErformance Evaluation 2
> TTC calulation based on camera with all possible combination of detector and descriptor is done and saved in **result_consolidated.csv** file.

> In my observation, SHITOMASI detector with BRISK or BRIEF provides good TTC results but as SHITOMASI is slower then we can use FAST detector if we can compromise few centimeters error in TTC calculations.

> In conclusion, I suggest use of Radar sensor in combination with lidar for TTC calculation will be faster and more robust.


