/*
MeanShiftImageSeg Examples

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

#include "opencv2/opencv.hpp"
#include <iostream>

#include "MeanShift.hpp"

#define MAX_PIXELS 200000

int main( int argc, char** argv )
{
   if(argc < 2)
   {
      std::cerr << "Usage: example[.exe] path/to/image/file" << std::endl;
      return -1;
   }

   cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

   if(image.data == NULL)
   {
      std::cerr << " - Error: couldn't open the input image file " << argv[1] << std::endl;
      return -1;
   }

   cv::Mat image3ch;
   if(image.channels() != 3 && image.channels() != 4)
   {
      std::cerr << " - Error: the input image must have 3 channels (" << image.channels() << " channels found!)" << std::endl;
      return -1;
   }
   else
   if(image.channels() == 4)
   {
      cvtColor(image, image3ch, CV_BGRA2BGR);
   }
   else
      image3ch = image;

   cv::Mat imageL1 = image3ch;
   while(imageL1.cols * imageL1.rows > MAX_PIXELS)
   {
      cv::Mat imageTemp;
      cv::pyrDown( imageL1, imageTemp, cv::Size( image3ch.cols/2, image3ch.rows/2 ) );
      imageL1 = imageTemp;
   }

   cv::Mat segmentedImage;
   MeanShift<5> meanShift;
   std::vector<MeanShift<5>::fsPoint> fsPointSet;
   meanShift.setInputImage(imageL1);
   meanShift.createFeatureSpace();
   meanShift.segment();
   meanShift.createSegmentedImage(segmentedImage);

   cv::imshow("imageL1", imageL1);
   cv::imshow("segmentedImage", segmentedImage);
   cv::waitKey();

   return 0;
}
