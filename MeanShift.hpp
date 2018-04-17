/*
MeanShiftImageSeg version 20180301

Copyright (c) 2018 Rasoul Mojtahedzadeh

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

#ifndef _MEAN_SHIFT_HPP_
#define _MEAN_SHIFT_HPP_

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include <vector>

template <int fsDim>
class MeanShift
{
public:
   typedef cv::Vec<float, fsDim> fsPoint;

   struct fsSegments
   {
     fsPoint mode;
     std::vector<size_t> index;

     fsSegments(fsPoint initMode, size_t initIndex)
     {
        mode = initMode;
        index.push_back(initIndex);
     }
   };

   MeanShift() :
      _sampleSetRadius(0.1f),
      _modeSquaredDistanceThreshold(0.05f),
      _modeSquaredChangeTolerance(0.0002f)
   {
      _sampleSetRadiusSquared = _sampleSetRadius * _sampleSetRadius;
      _sampleSetRadiusNegative = -_sampleSetRadius;
   }

   bool setInputImage(cv::Mat& inputImage)
   {
      if(inputImage.data == nullptr)
         return false;

      _inputImage = inputImage;
   }

   bool createFeatureSpace()
   {
      if(_inputImage.data == nullptr)
         return false;

      cv::Mat imageLuv;
      cv::cvtColor(_inputImage, imageLuv, CV_BGR2Luv);

      _fsPointSet.clear();
      for(int row = 0; row < imageLuv.rows; row++)
      {
         for(int col = 0; col < imageLuv.cols; col++)
         {
            cv::Vec3b Luv = imageLuv.at<cv::Vec3b>(row, col);
            if(fsDim == 5)
               _fsPointSet.push_back(fsPoint(Luv[0], Luv[1], Luv[2], row, col));
            else
            if(fsDim == 3)
               _fsPointSet.push_back(fsPoint(Luv[0], Luv[1], Luv[2]));
         }
      }

      return true;
   }

   bool getFsPointSet(std::vector<fsPoint> fsPointSet)
   {
      if(_fsPointSet.empty())
         return false;

      fsPointSet = _fsPointSet;

      return true;
   }

   bool createSegmentedImage(cv::Mat& segmentedImage)
   {
      if(_inputImage.data == nullptr)
         return false;

      if(_segments.empty())
         return false;

      segmentedImage.create(_inputImage.rows, _inputImage.cols, CV_8UC3);
      segmentedImage = cv::Scalar(0, 0, 0);

      cv::Mat Luv, bgr;
      Luv.create(1, 1, CV_8UC3);
      bgr.create(1, 1, CV_8UC3);
      for(size_t i = 0; i < _segments.size(); i++)
      {
         for(size_t j = 0; j < _segments[i].index.size(); j++)
         {
            const int row = _segments[i].index[j] / _inputImage.cols;
            const int col = _segments[i].index[j] % _inputImage.cols;
            Luv.at<cv::Vec3b>(0, 0) = cv::Vec3b(_segments[i].mode[0]*_maxims[0], _segments[i].mode[1]*_maxims[1], _segments[i].mode[2]*_maxims[2]);
            cv::cvtColor(Luv, bgr, CV_Luv2BGR);
            segmentedImage.at<cv::Vec3b>(row, col) = bgr.at<cv::Vec3b>(0, 0);
         }
      }

      return true;
   }

   bool setFeatureSpacePointSet(const std::vector<fsPoint>& fsPointSet)
   {
      if(fsPointSet.empty())
         return false;

      _fsPointSet = fsPointSet;

      return true;
   }

   bool segment()
   {
      if(!_normalizeFsPointSet()) return false;
      if(!_findModePoints()) return false;
      if(!_mergeModePoints()) return false;

      return true;
   }

   bool getSegments(std::vector<fsSegments>& segments)
   {
      if(_segments.empty())
         return false;

      segments = _segments;

      return true;
   }


private:
   float _sampleSetRadius;
   float _modeSquaredDistanceThreshold;
   float _modeSquaredChangeTolerance;

   float _sampleSetRadiusSquared;
   float _sampleSetRadiusNegative;

   cv::Mat _inputImage;

   std::vector<fsPoint>      _fsPointSet;
   std::vector<fsPoint>      _fsModePoints;
   std::vector<fsSegments>   _segments;
   std::vector<float>        _maxims;

   bool _normalizeFsPointSet()
   {
      _maxims.resize(fsDim);
      for(int j = 0; j < fsDim; j++)
         _maxims[j] = 0.f;

      for(size_t i = 0; i < _fsPointSet.size(); i++)
      {
         for(int j = 0; j < fsDim; j++)
         {
            if(_fsPointSet[i][j] > _maxims[j]) _maxims[j] = _fsPointSet[i][j];
         }
      }

      #pragma omp parallel for collapse(2)
      for(size_t i = 0; i < _fsPointSet.size(); i++)
      {
         for(int j = 0; j < fsDim; j++)
         {
            _fsPointSet[i][j] /= _maxims[j];
         }
      }

      return true;
   }

   bool _findModePoints()
   {
      _fsModePoints.resize(_fsPointSet.size());

      if(_fsModePoints.empty())
         return false;

      size_t fsPointIndex;
      #pragma omp parallel for private(fsPointIndex) schedule(dynamic) num_threads(8)
      for(fsPointIndex = 0; fsPointIndex < _fsPointSet.size(); fsPointIndex++)
      {
         fsPoint modePoint = _findMode(fsPointIndex);
         _fsModePoints[fsPointIndex] = modePoint;
      }

      return true;
   }

   bool _mergeModePoints()
   {
      if(_fsModePoints.empty())
         return false;

      for(size_t modePointIndex = 0; modePointIndex < _fsModePoints.size(); modePointIndex++)
      {
         bool found = false;
         for(size_t j = 0; j < _segments.size(); j++)
         {
            const float squaredDistanceSegmentMode = _squaredDistance(_segments[j].mode, _fsModePoints[modePointIndex]);
            if(squaredDistanceSegmentMode < _modeSquaredDistanceThreshold)
            {
               _segments[j].mode += (_fsModePoints[modePointIndex] - _segments[j].mode) / (float) _segments[j].index.size();
               _segments[j].index.push_back(modePointIndex);
               found = true;
               break;
            }
         }

         if(!found)
         {
            fsSegments segment(_fsModePoints[modePointIndex], modePointIndex);
            _segments.push_back(segment);
         }
      }
   }

   fsPoint _findMode(size_t index)
   {
      fsPoint modePoint = _fsPointSet[index];
      fsPoint meanPoint;

      float d[fsDim];

      while(1)
      {
         // --- Calculate mean point in radius r
         for(int j = 0; j < fsDim; j++)
         {
            meanPoint[j] = 0.0f;
         }

         size_t meanPointNum = 0;

         for(size_t i = 0; i < _fsPointSet.size(); i++)
         {
            float dSum = 0.0f;
            bool doBreak = false;
            for(int j = 0; j < fsDim; j++)
            {
               d[j] = modePoint[j] - _fsPointSet[i][j];
               if(d[j] > _sampleSetRadius || d[j] < _sampleSetRadiusNegative)
               {
                  doBreak = true;
                  break;
               }
            }

            if(doBreak)
            {
               continue;
            }

            for(int j = 0; j < fsDim; j++)
            {
               dSum += d[j] * d[j];
            }

            if(dSum < _sampleSetRadiusSquared)
            {
               meanPoint += _fsPointSet[i];
               meanPointNum++;
            }
         }

         if(meanPointNum > 0)
         {
            meanPoint /= (float) meanPointNum;
         }
         else
         {
            meanPoint = modePoint;
            return meanPoint;
         }

         const float squaredDistanceMeanMode = _squaredDistance(meanPoint, modePoint);
         if(squaredDistanceMeanMode < _modeSquaredChangeTolerance)
         {
            return meanPoint;
         }
         else
         {
            modePoint = meanPoint;
         }
      };
   }

   float _squaredDistance(const fsPoint& p1, const fsPoint& p2)
   {
      float dSum = 0.0f;
      for(int j = 0; j < fsDim; j++)
      {
         const float dj = p1[j] - p2[j];
         dSum += dj * dj;
      }

      return dSum;
   }
};

#endif // _MEAN_SHIFT_HPP_
