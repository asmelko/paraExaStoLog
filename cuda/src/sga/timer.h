// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

/*
GAP Benchmark Suite
Class:  Timer
Author: Scott Beamer

Simple timer that wraps gettimeofday
*/


class Timer
{
public:
	Timer() {}

	void Start() { start = std::chrono::steady_clock::now(); }

	void Stop() { stop = std::chrono::steady_clock::now(); }

	double Seconds() const { return std::chrono::duration_cast<std::chrono::seconds>(stop - start).count(); }

	double Millisecs() const { return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); }

	double Microsecs() const { return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count(); }

private:
	std::chrono::steady_clock::time_point start, stop;
};

// Times op's execution using the timer t
#define TIME_OP(t, op)                                                                                                 \
	{                                                                                                                  \
		t.Start();                                                                                                     \
		(op);                                                                                                          \
		t.Stop();                                                                                                      \
	}

#endif // TIMER_H_
