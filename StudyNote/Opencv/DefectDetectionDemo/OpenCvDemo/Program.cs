﻿using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace OpenCvDemo
{
    class Program
    {
        static void Main()
        {
            var input = new List<Demo>
            {
                new Demo {File = "../../../Image/Dirty.jpg", Blur = 7, ThresholdLow = 10, ThresholdHigh = 30},
                new Demo {File = "../../../Image/Oil.jpg", Blur = 11, ThresholdLow = 9, ThresholdHigh = 26, Filter = true},
                new Demo {File = "../../../Image/Line.jpg", Blur = 11, ThresholdLow = 10, ThresholdHigh = 25, Filter = true},
                new Demo {File = "../../../Image/test1.bmp", Blur = 7, ThresholdLow = 10, ThresholdHigh = 30},
                new Demo {File = "../../../Image/test2.bmp", Blur = 11, ThresholdLow = 9, ThresholdHigh = 26, Filter = true},
                new Demo {File = "../../../Image/test3.bmp", Blur = 11, ThresholdLow = 10, ThresholdHigh = 25, Filter = true}
            };

            Show(input[0]);
        }

        public static void Show(Demo demo)
        {
            var org = new Mat();
            var img = Cv2.ImRead(demo.File);

            //var img = new Mat();
            //Cv2.Resize(org_img, img, new OpenCvSharp.Size(224, 224));

            var gray = new Mat(demo.File, ImreadModes.GrayScale);
            //var gray  =  new Mat();
            //Cv2.Resize(gray1, gray, new OpenCvSharp.Size(224, 224));

            img.CopyTo(org);

            #region blur
            var blur = new Mat();
            Cv2.GaussianBlur(gray, blur, new Size(demo.Blur, demo.Blur), 0);
            #endregion

            #region canny
            var canny = new Mat();
            Cv2.Canny(blur, canny, demo.ThresholdLow, demo.ThresholdHigh);
            Cv2.Dilate(canny, canny, new Mat());
            #endregion

            #region contours
            Point[][] contours;
            HierarchyIndex[] hierarchyIndexes;
            Cv2.FindContours(
                canny,
                out contours,
                out hierarchyIndexes,
                mode: RetrievalModes.External,
                method: ContourApproximationModes.ApproxSimple);
            #endregion

            if(demo.Filter) contours = contours.Where(x => x.Length > 40).ToArray();
            Cv2.DrawContours(img, contours, -1, Scalar.Red, thickness: 2);

            using (new Window("image", img))
            using (new Window("org", org))
            {
                Cv2.WaitKey();
            }
        }
    }

    public class Demo
    {
        public string File { get; set; }
        public int Blur { get; set; }
        public int ThresholdLow { get; set; }
        public int ThresholdHigh { get; set; }
        public bool Filter { get; set; }
    }
}
