/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#pragma once

#include "util/settings.h"
#include "boost/thread.hpp"
#include <stdio.h>
#include <iostream>


namespace dso {
// was invoked in fullsystem:
// IndexThreadReduce<Vec10> treadReduce;
// typename "Running" here is a container?
// typedef Eigen::Matrix<double,10,1> Vec10 -> inside NumType.h
    template<typename Running>
    class IndexThreadReduce {

    public:
        /*
         * https://github.com/stack-of-tasks/pinocchio/issues/165
         * To be able to instantiate a model and data using the new operator one should add
         * EIGEN_MAKE_ALIGNED_OPERATOR_NEW
         * as a public macro in the classes.
         * This is needed feature if you want to have pointers own and initialized in a single class.
         * */
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // constructor
        inline IndexThreadReduce() {
            // nextIndex and maxIndex was used in workerLoop
            // workerLoop seems like a working thread
            nextIndex = 0;
            maxIndex = 0;
            stepSize = 1;
            // this literally did nothing >>>.....
            callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);

            running = true;
            // num_threads was set to 6, they initialize those 6 worker threads.
            for (int i = 0; i < NUM_THREADS; i++) {
                isDone[i] = false;
                gotOne[i] = true;
                workerThreads[i] = boost::thread(&IndexThreadReduce::workerLoop, this, i);
            }

        }

        inline ~IndexThreadReduce() {
            running = false;

            exMutex.lock();
            todo_signal.notify_all();
            exMutex.unlock();

            for (int i = 0; i < NUM_THREADS; i++)
                workerThreads[i].join();


            printf("destroyed ThreadReduce\n");

        }

        // use reduce function to multi threads a given task
        inline void
        reduce(boost::function<void(int, int, Running *, int)> callPerIndex, int first, int end, int stepSize = 0) {
            // initialize the stats as all 0 -> vec10 is intialized as 0
            memset(&stats, 0, sizeof(Running));

//		if(!multiThreading)
//		{
//			callPerIndex(first, end, &stats, 0);
//			return;
//		}



            if (stepSize == 0)
                stepSize = ((end - first) + NUM_THREADS - 1) / NUM_THREADS;


            //printf("reduce called\n");

            boost::unique_lock<boost::mutex> lock(exMutex);
            // this is the reason why they will call eigen memory align macro at first
            // they want to save the pointer owned in the class
            // save
            this->callPerIndex = callPerIndex;
            nextIndex = first;
            maxIndex = end;
            this->stepSize = stepSize;

            // go worker threads!
            for (int i = 0; i < NUM_THREADS; i++) {
                isDone[i] = false;
                gotOne[i] = false;
            }

            // let them start!
            todo_signal.notify_all();


            //printf("reduce waiting for threads to finish\n");
            // wait for all worker threads to signal they are done.
            while (true) {
                // wait for at least one to finish
                done_signal.wait(lock);
                //printf("thread finished!\n");

                // check if actually all are finished.
                bool allDone = true;
                for (int i = 0; i < NUM_THREADS; i++)
                    allDone = allDone && isDone[i];

                // all are finished! exit.
                if (allDone)
                    break;
            }

            nextIndex = 0;
            maxIndex = 0;
            this->callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);

            //printf("reduce done (all threads finished)\n");
        }

        // status is the instance of, say, vec10.
        Running stats;

    private:
        boost::thread workerThreads[NUM_THREADS];
        bool isDone[NUM_THREADS];
        bool gotOne[NUM_THREADS];

        boost::mutex exMutex;
        boost::condition_variable todo_signal;
        boost::condition_variable done_signal;

        int nextIndex;
        int maxIndex;
        int stepSize;

        bool running;

        // boost::function is a function pointer that point to certain pattern:
        // the parameter and return types are specified like this:
        // <void(int,int,Running*,int)> -> return void, take three int and one Running* arguments
        boost::function<void(int, int, Running *, int)> callPerIndex;

        void callPerIndexDefault(int i, int j, Running *k, int tid) {
            printf("ERROR: should never be called....\n");
            assert(false);
        }

        void workerLoop(int idx) {
            boost::unique_lock<boost::mutex> lock(exMutex);

            while (running) {
                // try to get something to do.
                int todo = 0;
                bool gotSomething = false;
                if (nextIndex < maxIndex) {
                    // got something!
                    todo = nextIndex;
                    nextIndex += stepSize;
                    gotSomething = true;
                }

                // if got something: do it (unlock in the meantime)
                if (gotSomething) {
                    lock.unlock();

                    assert(callPerIndex != 0);

                    Running s;
                    memset(&s, 0, sizeof(Running));
                    callPerIndex(todo, std::min(todo + stepSize, maxIndex), &s, idx);
                    gotOne[idx] = true;
                    lock.lock();
                    stats += s;
                }

                    // otherwise wait on signal, releasing lock in the meantime.
                else {
                    if (!gotOne[idx]) {
                        lock.unlock();
                        assert(callPerIndex != 0);
                        Running s;
                        memset(&s, 0, sizeof(Running));
                        callPerIndex(0, 0, &s, idx);
                        gotOne[idx] = true;
                        lock.lock();
                        stats += s;
                    }
                    isDone[idx] = true;
                    //printf("worker %d waiting..\n", idx);
                    done_signal.notify_all();
                    todo_signal.wait(lock);
                }
            }
        }
    };
}
