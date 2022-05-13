#include <ctime>
#include "cgp.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "Image4CGP.h"
#include <filesystem>

#include <bits/stdc++.h>
#include <boost/algorithm/string.hpp>


using namespace cv;
using namespace std;
namespace fs = std::filesystem;


bool files_comparator(pair<int, struct chromosome*> p1, pair<int, struct chromosome*> p2) {
    return p1.first < p2.first;
}

int recovering(struct parameters* params, const string& path2srcimages,
               string& path2chromos, int extIterNum, bool logging = false, bool show_new_pictures = false){

    const int numImages = getNumImages(params);
    const int width = getWidth(params);
    const int height = getHeight(params);
    const int numRes = getImageResolution(params);
    vector<string> file_names;
    struct dataSet* data;
    vector<struct chromosome*> chromos;
    vector<pair<int, struct chromosome*>> chromos_pairs;
    struct chromosome* chromo;
    auto* recovered = new double[numImages * numRes];
    string name;
    auto* imageArray = new double[width*height];

    char *fname;
    string file_name;
    regex regexp("[0-9]+");
    smatch m;

    for (const auto & entry: fs::directory_iterator(path2chromos)) {
        file_name = entry.path().string();
        fname = strcpy((char*)malloc(file_name.length()+1), file_name.c_str());
        regex_search(file_name, m, regexp);
        chromo = initialiseChromosomeFromFile(fname, numImages);
        chromos_pairs.push_back(make_pair(stoi(m[0]), chromo));
        chromos.push_back(chromo);
        delete [] fname;
    }

    sort(chromos_pairs.begin(), chromos_pairs.end(), files_comparator);

    data = loadDataSetFromImages(path2srcimages, file_names, numImages,
                                         width, height,
                                         true, true);

    const double MAX_VALUE = pow(10, 10);
    const double EPSILON = pow(10, -15);
    double meanImage;
    double meanPred = 0;
    double meanImage2;
    double meanPred2 = 0;
    double meanImagePred; // the value Image*Prediction
    double varianceImage;
    double variancePred = 0;
    double covariance;
    double a, b;
    auto* imagePred = new double[numRes];

    auto* inputs = new double[2 * numImages * numRes];
    auto* outputs = new double[numImages * numRes];

    string prefix = string("rec_");
    for (int i = 0; i < numImages; i++) {
        for (int p = 0; p < numRes; p++) {
            inputs[2 * i * numRes + 2 * p] = double(p / width);
            inputs[2 * i * numRes + 2 * p + 1] = double(p % width);
            recovered[i * numRes + p] = 0;
        }
    }

    double recovered_pixel = 0;
    double mp_trash=0, mp2_trash=0;
    int infinite_pred_statistics = 0;
    int infinite_common_statistics = 0;

    for (int ei = 0; ei<extIterNum; ei++){
        if (ei % 50 == 0)
            cout << "Iteration: " << ei << endl;
        chromo = chromos_pairs[ei].second;

        mp_trash=0;
        mp2_trash=0;
        meanPred = 0;
        meanPred2 = 0;

        for (int p=0; p<numRes; p++) {
            executeChromosome(chromo, getDataSetSampleInputs(data, p));
            double pixel = getChromosomeOutput(chromo,0);
            imagePred[p] = pixel;
            meanPred += pixel;
            meanPred2 += pixel * pixel;

            mp_trash += pixel / numRes;
            mp2_trash += pixel * pixel / numRes;
        }

        meanPred /= numRes;
        meanPred2 /= numRes;
        variancePred = meanPred2 - meanPred * meanPred;

        if (!isfinite(variancePred)) {
            cout << "Any problems with variancePred: \n\tvariancePred=" << variancePred
            << "; meanPred2=" << meanPred2 << "; meanPred=" << meanPred << endl;
            cout << "But: \n\tmp_trash=" << mp_trash << "; mp2_trash=" << mp2_trash << endl;
            infinite_pred_statistics += 1;
            continue;
        }

        for (int i=0; i<numImages; i++){
            meanImage = 0;
            meanImage2 = 0;
            meanImagePred = 0;

//            #pragma omp parallel for default(none), shared(meanImage,meanImage2,meanImagePred,i,numRes,imagePred,data,params), schedule(dynamic), num_threads(getNumThreads(params))
            for (int p=0; p<numRes; p++) {
                double pixel = getDataSetSampleOutput(data, i*numRes + p, 0);
                meanImage += pixel;
                meanImage2 += pixel * pixel;
                meanImagePred += pixel * imagePred[p];
            }
            meanImage /= numRes;
            meanImage2 /= numRes;
            varianceImage = meanImage2 - meanImage * meanImage;
            covariance = (meanImagePred - numRes * meanImage * meanPred) / (numRes - 1); // /varianceImage/variancePred;

            if (!isfinite(variancePred) or !isfinite(covariance / variancePred)) {
                cout << "Oops!:\n\t" << "variancePred=" << variancePred << "; covariance=" << covariance << "; covariance / variancePred=" << covariance / variancePred << endl;
                setA(chromo, i, 0);
                setB(chromo, i, 0);
//                break;
            } else {
                // save A and B coefficients (linear scaling)
                b = covariance / variancePred;
                a = meanImage - b*meanPred;
                setA(chromo, i, a);
                setB(chromo, i, b);
            }

            a = getA(chromo, i);
            b = getB(chromo, i);

//            #pragma omp parallel for default(none), shared(recovered_pixel,a,b,i,numRes,imagePred,data,outputs,recovered,imageArray), schedule(dynamic), num_threads(getNumThreads(params))
            for (int p=0; p<numRes; p++) {
                recovered_pixel = a + b*imagePred[p];
                outputs[i * numRes + p] = getDataSetSampleOutput(data, i*numRes + p,0) - recovered_pixel;
                recovered[i * numRes + p] += recovered_pixel;
                imageArray[p] = recovered[i * numRes + p];
            }

            if (ei > 0 and ei % 1000 == 0) {
                cv::Mat greyImg = cv::Mat(height, width, CV_64F, imageArray);
                std::string greyArrWindow = String("Image") + to_string(i);
                cv::namedWindow(greyArrWindow, cv::WINDOW_NORMAL);
                cv::imshow(greyArrWindow, greyImg);
                waitKey(0);
            }
        }
        freeDataSet(data);
        data = initialiseDataSetFromArrays(2, 1, numImages * numRes, inputs, outputs);
    }

    for (int i = 0; i < numImages; i++) {
        for (int p=0; p < numRes; p++) {
            imageArray[p] = recovered[i * numRes + p];
        }
        cv::Mat greyImg = cv::Mat(height, width, CV_64F, imageArray);
        greyImg *= 255.0;
        name = string("../RecoveryResult/") + prefix + file_names[i] + ".png";
        imwrite(name, greyImg);

        if (show_new_pictures) {
            std::string greyArrWindow = String("Image") + to_string(i);
            cv::namedWindow(greyArrWindow, cv::WINDOW_NORMAL);
            cv::imshow(greyArrWindow, greyImg / 255);
            waitKey(0);
        }
    }

    delete [] recovered;
    freeDataSet(data);
    // TODO: free chomosomes

    cout << "infinite_pred_statistics: " << infinite_pred_statistics << endl;
    cout << "infinite_common_statistics: " << infinite_common_statistics << endl;

    return 0;
}


int main() {
    const int ext_iterNum = 6000;
    const int width = 256;
    const int height = 256;
    const int numImages = 5;
    const int numThreads = 4;
    string path2srcimages = string("../images2gpfl_256");
    string path2chromos = string("../Chromosomes");

    int numInputs = 2;
    int numNodes = 300;
    int numOutputs = 1;
    int nodeArity = 2;

    struct parameters* params;
    params = initialiseParameters(numInputs, numNodes, numOutputs, nodeArity);
    //    important detail for GPFL
    setNumImages(params, numImages);
    setImageResolution(params, width*height);
    setWidth(params, width);
    setHeight(params, height);
    setNumThreads(params, 4);

    recovering(params, path2srcimages, path2chromos, ext_iterNum, false, true);

    return 0;
}
