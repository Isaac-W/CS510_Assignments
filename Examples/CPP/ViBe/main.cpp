#include "bgfg_vibe.hpp"
using namespace cv;

int main(int argc, char ** argv)
{
    Mat frame;
    bgfg_vibe bgfg;
    if ( argc < 2)
    {
        printf("usage: %s <image or movie file>", argv[0]);
        return 1;
    }
    VideoCapture cap(argv[1]);
    cap >> frame;
    bgfg.init_model(frame);
    for(;;)
    {
        cap>>frame;
        Mat fg = *bgfg.fg(frame);
        imshow("fg",fg);
        waitKey(1);
    }
    return 0;
}
