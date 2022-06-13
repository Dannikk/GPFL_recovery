#include <ctime>
#include "cgp.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "Image4CGP.h"
#include "EdgeDetection.h"
#include <filesystem>
#include <chrono>
#include <omp.h>

#include <bits/stdc++.h>
#include <boost/algorithm/string.hpp>
#include <omp.h>


using namespace cv;
using namespace std;
using namespace std::chrono;
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
    chromos_pairs.reserve(extIterNum);
    for (const auto & entry: fs::directory_iterator(path2chromos)) {
//        cout << "chromos_pairs.size(): " << chromos_pairs.size()<< "; chromos_pairs.capacity(): " << chromos_pairs.capacity() << endl;
        file_name = entry.path().string();
        fname = strcpy((char*)malloc(file_name.length()+1), file_name.c_str());
        regex_search(file_name, m, regexp);
        chromo = initialiseChromosomeFromFile(fname, numImages);
        chromos_pairs.emplace_back(stoi(m[0]), chromo);
        delete [] fname;
    }

    sort(chromos_pairs.begin(), chromos_pairs.end(), files_comparator);

    data = loadDataSetFromImages(path2srcimages, file_names, numImages,
                                         width, height,
                                         true, false);

    const double MAX_VALUE = pow(10, 10);
    const double EPSILON = pow(10, -15);
//    double meanImage;
    double meanPred = 0;
//    double meanImage2;
    double meanPred2 = 0;
//    double meanImagePred; // the value Image*Prediction
//    double varianceImage;
    double variancePred = 0;
//    double covariance;
//    double a, b;
    auto* imagePred = new double[numRes];

    auto* inputs = new double[2 * numImages * numRes];
    auto* outputs = new double[numImages * numRes];

    string prefix = string("rec_");

    for (int i = 0; i < numImages; i++) {
        for (int p = 0; p < numRes; p++) {
            inputs[2 * i * numRes + 2 * p] = double(p / width);
            inputs[2 * i * numRes + 2 * p + 1] = double(p % width);
            outputs[i*numRes + p] = getDataSetSampleOutput(data, i*numRes + p, 0);
            recovered[i * numRes + p] = 0;
        }
    }

//    double recovered_pixel = 0;
    double mp_trash=0, mp2_trash=0;
    int infinite_pred_statistics = 0;
    int infinite_common_statistics = 0;

    FILE *fp = fopen("../coeffs.txt", "w");

    for (int i=0; i < numImages; i++) {
        fprintf(fp, "%s\t\t\t\t", file_names[i].c_str());
    }
    fprintf(fp, "\n");

    //
    auto start = high_resolution_clock::now();
    for (int ei = 0; ei<extIterNum; ei++) {
        if (ei % 100 == 0)
            cout << "Iteration: " << ei << endl;
        chromo = chromos_pairs[ei].second;
        /*if (chromo == nullptr)
            cout << "Here null chromo: " << ei << endl;*/

        mp_trash=0;
        mp2_trash=0;
        meanPred = 0;
        meanPred2 = 0;

//        auto start = high_resolution_clock::now();
        struct chromosome** chromos_par = (struct chromosome**)malloc(sizeof(struct chromosome*)*4);
        double pmean_0=0, pmean_1=0, pmean_2=0, pmean_3=0;
        double pmean2_0=0, pmean2_1=0;
        for(int i=0; i < 4; i++){
            chromos_par[i] = initialiseChromosomeFromChromosome(chromo);
//            pred_means[i] = 0;
        }
//        #pragma omp parallel for default(none), shared(numRes,meanPred,meanPred2,chromos_par,data,imagePred), schedule(dynamic), num_threads(2)
//        for (int p=0; p<numRes; p++) {
//            executeChromosome(chromos_par[omp_get_thread_num()], getDataSetSampleInputs(data, p));
//            double pixel = getChromosomeOutput(chromos_par[omp_get_thread_num()],0);
//            imagePred[p] = pixel;
//            meanPred += pixel;
//            meanPred2 += pixel * pixel;
//        }

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                double pixel;
                for(int p=0; p < numRes/2; p++){
                    pixel = executeChromosome(chromos_par[0], getDataSetSampleInputs(data, p));
                    imagePred[p] = pixel;
                    pmean_0 += pixel;
                    pmean2_0 += pixel * pixel;
                }
            }
            #pragma omp section
            {
                double pixel;
                for(int p=numRes/2; p < numRes; p++){
                    pixel = executeChromosome(chromos_par[1], getDataSetSampleInputs(data, p));
                    imagePred[p] = pixel;
                    pmean_1 += pixel;
                    pmean2_1 += pixel * pixel;
                }
            }
        }

        for(int i=0; i < 4; i++){
            free(chromos_par[i]);
        }
        delete [] chromos_par;

        /*auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "predict stats calcing: " << duration.count() << endl;*/

        meanPred += pmean_0 + pmean_1;
        meanPred2 += pmean2_0 + pmean2_1;

        meanPred /= numRes;
        meanPred2 /= numRes;
        variancePred = meanPred2 - meanPred * meanPred;

        if (!isfinite(variancePred)) {
            cout << "\tOn iteration " << ei << endl;
            cout << "Any problems with variancePred: \n\tvariancePred=" << variancePred
            << "; meanPred2=" << meanPred2 << "; meanPred=" << meanPred << endl;
            cout << "But: \n\tmp_trash=" << mp_trash << "; mp2_trash=" << mp2_trash << endl;
            infinite_pred_statistics++;
            continue;
        }

//        start = high_resolution_clock::now();
        //#pragma omp parallel for default(none), shared(ei,numRes,numImages,imagePred,meanPred,variancePred,chromo,data,height,width,show_new_pictures,outputs,recovered,imageArray,infinite_common_statistics), schedule(dynamic), num_threads(getNumThreads(params))
        for (int i=0; i<numImages; i++){
            double meanImage = 0;
            double meanImage2 = 0;
            double meanImagePred = 0;

//            #pragma omp parallel for default(none), shared(meanImage,meanImage2,meanImagePred,i,numRes,imagePred,data,params), schedule(dynamic), num_threads(getNumThreads(params))
            for (int p=0; p<numRes; p++) {
//                double pixel = getDataSetSampleOutput(data, i*numRes + p, 0);
                double pixel = outputs[i*numRes + p];
                meanImage += pixel;
                meanImage2 += pixel * pixel;
                meanImagePred += pixel * imagePred[p];
            }
            meanImage /= numRes;
            meanImage2 /= numRes;
            double varianceImage = meanImage2 - meanImage * meanImage;
            double covariance = (meanImagePred - numRes * meanImage * meanPred) / (numRes - 1); // /varianceImage/variancePred;

            if (!isfinite(variancePred) or !isfinite(covariance / variancePred)) {
//                cout << "Oops!:\n\t" << "variancePred=" << variancePred << "; covariance=" << covariance << "; covariance / variancePred=" << covariance / variancePred << endl;
                setA(chromo, i, 0);
                setB(chromo, i, 0);
                infinite_common_statistics++;
//                break;
            } else {
                // save A and B coefficients (linear scaling)
                double b = covariance / variancePred;
                double a = meanImage - b*meanPred;
                setA(chromo, i, a);
                setB(chromo, i, b);
            }

            double a = getA(chromo, i);
            double b = getB(chromo, i);

//            #pragma omp parallel for default(none), shared(a,b,i,numRes,imagePred,data,outputs,recovered,imageArray), schedule(dynamic), num_threads(getNumThreads(params))
            for (int p=0; p<numRes; p++) {
                double recovered_pixel = a + b*imagePred[p];
                outputs[i * numRes + p] = outputs[i * numRes + p] - recovered_pixel;
                recovered[i * numRes + p] += recovered_pixel;
                imageArray[p] = recovered[i * numRes + p];
            }

            if (show_new_pictures) {
                if (ei > 0 and ei % 1000 == 0) {
                    cv::Mat greyImg = cv::Mat(height, width, CV_64F, imageArray);
                    std::string greyArrWindow = String("Image") + to_string(i);
                    cv::namedWindow(greyArrWindow, cv::WINDOW_NORMAL);
                    cv::imshow(greyArrWindow, greyImg);
                    waitKey(0);
                }
            }
        }

        /*stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << "recovering calcing: " << duration.count() << endl;*/

        for (int i=0; i < numImages; i++) {
            fprintf(fp, "%.15E %.15E\t", getA(chromo, i), getB(chromo, i));
        }
        fprintf(fp, "\n");
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "recovering time: " << duration.count() << endl;

    cout << "Reverse recovering" << endl;
    double** reverseRecovering = new double*[numImages];
    for(int i=0; i<numImages; i++) {
        reverseRecovering[i] = new double[width*height];
        for(int p=0; p<numRes; p++) {
            reverseRecovering[i][p] = 0;
        }
    }

    for(int ei=extIterNum-1; ei >= 0; ei--) {
        chromo = chromos_pairs[ei].second;
        if ((ei + 1) % 500 == 0)
            cout << "Reverse iteration " << ei+1 << endl;
        for(int p=0; p<numRes; p++) {
            executeChromosome(chromo, getDataSetSampleInputs(data, p));
            double pix = getChromosomeOutput(chromo, 0);
            for (int i = 0; i < numImages; i++) {
                double a = getA(chromo, i);
                double b = getB(chromo, i);
                reverseRecovering[i][p] += a + b*pix;
            }
        }

        if (ei != (extIterNum-1) and (ei+1) % 500 == 0 or ei==0){
            for(int i=0; i<numImages; i++) {
                cv::Mat greyImg = cv::Mat(height, width, CV_64F, reverseRecovering[i]);
                name = string("../ReverseRecovering/") + to_string(ei + 1) + string("_") + file_names[i] + ".png";
                imwrite(name, greyImg * 255);
            }
        }
    }

    fclose(fp);

    for (int i = 0; i < numImages; i++) {
        if(i==0){
            FILE* rec_pixels = fopen("../rec_pixels.txt", "w");
            for (int p = 0; p < numRes; p++) {
                imageArray[p] = recovered[i * numRes + p];
                fprintf(rec_pixels, "%.30f\n", imageArray[p]);
            }
            fclose(rec_pixels);
        }
        else {
            for (int p = 0; p < numRes; p++) {
                imageArray[p] = recovered[i * numRes + p];
            }
        }
        cv::Mat greyImg = cv::Mat(height, width, CV_64F, imageArray);
//        greyImg *= 255.0;
        name = string("../RecoveryResult/") + prefix + file_names[i] + ".png";
        imwrite(name, greyImg * 255);

        if (show_new_pictures) {
            std::string greyArrWindow = String("Image") + to_string(i);
            cv::namedWindow(greyArrWindow, cv::WINDOW_NORMAL);
            cv::imshow(greyArrWindow, greyImg);
            waitKey(0);
        }
    }

    delete [] recovered;
    freeDataSet(data);
    // TODO: free chomosomes

    cout << "infinite_pred_statistics: " << infinite_pred_statistics << endl;
    cout << "infinite_common_statistics: " << infinite_common_statistics << endl;

    auto** edges = new double*[numImages];
    auto* gradients = new double[numImages];

    for(int i=0; i < numImages; i++) {
        edges[i] = new double[width * height];
    }

    chromos.reserve(extIterNum);
    for(int ei=0; ei<extIterNum; ei++) {
        chromos.push_back(chromos_pairs[ei].second);
    }

    cout << "chromos.size(): " << chromos.size()<< "chromos.capacity()" << chromos.capacity() << endl;

    const double epsilon = pow(10, -2);

    auto minimums = new double[numImages];
    auto maximums = new double[numImages];
    for (int i=0; i< numImages; i++){
        minimums[i] = edges[i][0];
        maximums[i] = edges[i][0];
    }
    double pixel;
    double mean_grad_0=0;
    int infinite_grad_counter = 0;

    FILE* pixels = fopen("../pixels.txt", "w");

    for(int p=0; p<numRes; p++) {
        if (p % 256 == 0 and (p / 256) % 20 == 0)
            cout << "x=" << (p / 256) << "; " << p % 256 << endl;

//        if ((p % 256) % 100 == 0 and (p / 256) % 100 == 0)
        if (false)
            gradient_module(p / width, p % width, gradients, extIterNum, params, chromos, epsilon, true);
        else
            gradient_module(p / width, p % width, gradients, extIterNum, params, chromos, epsilon, false);
        if (isfinite(gradients[0])) {
            mean_grad_0 += gradients[0];
        } else {
            infinite_grad_counter++;
        }
        for(int i=0; i<numImages; i++){
            fprintf(pixels, "%.30f ", gradients[i]);
            pixel = gradients[i];
            edges[i][p] = pixel;
            if (pixel<minimums[i])
                minimums[i] = pixel;
            if (pixel>maximums[i])
                maximums[i]=pixel;
        }
        fprintf(pixels, "\n");
    }
    fclose(pixels);

    /*for(int x=0; x<width; x++) {
        if (x % 20 == 0)
            cout << "x=" << x << endl;
        for (int y=0; y<height; y++) {
            gradient_module(x, y, gradients, extIterNum, params, chromos, epsilon);
            if (isfinite(gradients[0])) {
                fprintf(pixels, "%.30f\n", gradients[0]);
                mean_grad_0 += gradients[0];
            } else {
                infinite_grad_counter++;
            }
            for(int i=0; i<numImages; i++){
                pixel = gradients[i];
                edges[i][x + y*width] = pixel;
                if (pixel<minimums[i])
                    minimums[i] = pixel;
                if (pixel>maximums[i])
                    maximums[i]=pixel;
            }
        }
    }*/

    cout << "mean of first image grad: " << mean_grad_0 / (width * height - infinite_grad_counter) << endl;
    cout << "infinite_grad_counter: " << infinite_grad_counter << endl;

    cout << "min and max: " << endl;
    for(int i=0; i < numImages; i++){
        cout << "min: " << minimums[i] << "; max: " << maximums[i] << endl;
    }

    prefix = string("edge_");

    for (int i = 0; i < numImages; i++) {
        cv::Mat greyImg = cv::Mat(height, width, CV_64F, edges[i]);
//        name = string("../EdgeDetection/") + prefix + file_names[i] + ".png";
//        imwrite(name, greyImg);

        name = string("../EdgeDetection/") + prefix + string("255") + file_names[i] + ".png";
        imwrite(name, greyImg*255);
    }

    return 0;
}


int main() {
    const int ext_iterNum = 6000;
    const int width = 256;
    const int height = 256;
    const int numImages = 5;
    const int numThreads = 4;
    string path2srcimages = string("../images2gpfl_256");
    string path2chromos = string("../Chromosomes_6000");

    /*cout << "creating circle" << endl;
    auto* ccc = new double[width*height];
    for(int x=0; x<width; x++){
        for(int y=0; y<height; y++){
            ccc[y*width + x] = abs((x-width/2)*(y-height/2)) / 100;
        }
    }

    cout << "saving circle" << endl;
    cv::Mat greyImg = cv::Mat(height, width, CV_64F, ccc);
    imwrite("../circle.png", greyImg);
    imwrite("../circle_255.png", greyImg*255);*/

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
    setNumThreads(params, numThreads);

    recovering(params, path2srcimages, path2chromos, ext_iterNum, false, false);

    return 0;
}
