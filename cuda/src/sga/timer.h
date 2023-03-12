// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef TIMER_H_
#define TIMER_H_



/*
GAP Benchmark Suite
Class:  Timer
Author: Scott Beamer

Simple timer that wraps gettimeofday
*/


class Timer {
 public:
  Timer() {}

  void Start() {
  }

  void Stop() {
  }

  double Seconds() const { return 0.;
  }

  double Millisecs() const { return 0.;
  }

  double Microsecs() const { return 0.;
  }

 private:
};

// Times op's execution using the timer t
#define TIME_OP(t, op) { t.Start(); (op); t.Stop(); }

#endif  // TIMER_H_
