#include "mateer_gao.h"

void log_process(uint64_t interval_start, uint64_t interval_length, unsigned int interval_pow2size)
{
  cout << dec << "start : " << interval_start << " ; length : " << interval_length << " ; power of 2 bound of size : " << interval_pow2size << endl;
}


//depth-first tree search, in non-recursive form. call a function on all subtrees (represented by intervals)
//with size > 2**logtau
void DFTS(unsigned int logtau, unsigned int logsize, function<void(uint64_t, uint64_t, unsigned int)> f)
{
  uint64_t interval_start = 0;
  unsigned int interval_logsize = logsize;
  if(interval_logsize <= logtau) return;
  int interval_state = 0 ;// 0: pas fait ; 1: moitié gauche faite; 2 : fait
  while(true)
  {
    uint64_t interval_length = 1uL << interval_logsize;
    bool is_right_subinterval = interval_start & interval_length;
    switch(interval_state)
    {
    case 0:
      f(interval_start, 1uL << interval_logsize, interval_logsize);
      if(interval_logsize > 1+logtau)
      {
        // explore left subtree
        interval_logsize--;
      }
      else
      {
        // arrived at a leaf: determine if it is a left or right leaf (or the full interval is a leaf)
        if(interval_logsize == logsize) return;
        if(is_right_subinterval)
        {
          interval_start -= interval_length;
          interval_logsize++;
          interval_state = 2;
        }
        else
        {
          // jump directly to right leaf at same level
          interval_start += interval_length;
          interval_state = 0;
        }
      }
      break;
    case 1:
      // we can't be at a leaf here: explore right subtree
      interval_logsize--;
      interval_start += 1uL << interval_logsize;
      interval_state = 0;
      break;
    case 2:
      cout << "Finished processing subtree starting at " << interval_start << " and of size " << interval_length << endl;
      if(interval_logsize == logsize) return;
      else
      {
        // go up, adjusting interval start if on the right subtree
        if(is_right_subinterval)
        {
          interval_start -= 1uL << interval_logsize;
          interval_logsize++;
          interval_state = 2;
        }
        else
        {
          interval_logsize++;
          interval_state = 1;
        }
      }
      break;
    }
  }
}


//depth-first tree search, non-recursive form, for an incomplete tree
void DFTS_incomplete_tree(
    unsigned int logtau,
    unsigned int logsize,
    uint64_t num_elts,
    function<void(uint64_t, uint64_t, unsigned int)> f)
{
  uint64_t interval_start = 0;
  unsigned int interval_logsize = logsize;
  if(interval_logsize <= logtau) return;
  if(num_elts == 0) return;
  assert (num_elts <= (1uL << interval_logsize) && num_elts > (1uL << (interval_logsize - 1)));
  int interval_state = 0 ;// 0: pas fait ; 1: moitié gauche faite; 2 : fait
  uint64_t alt_start = 0;
  while(true)
  {
    // length of the current interval (the one 'interval_state' applies to)
    uint64_t interval_length = 1uL << interval_logsize;
    uint64_t actual_length = min(num_elts - interval_start, interval_length);
    bool is_right_subinterval = interval_start & interval_length;
    switch(interval_state)
    {
    case 0:
      f(interval_start, actual_length, interval_logsize);
      if(interval_logsize > 1+logtau)
      {
        // explore left subtree
        interval_logsize--;
      }
      else
      {
        // current interval is a leaf: determine if it is a left or right leaf
        // go up if it is a right leaf or (if it is a left leaf and right leaf is off limits)
        if(interval_logsize == logsize) return;
        if(is_right_subinterval || (interval_start + interval_length) >= num_elts)
        {
          //go up
          if(is_right_subinterval) interval_start -= interval_length;
          interval_logsize++;
          interval_state = 2;
        }
        else
        {
          // jump directly to right leaf at same level
          // interval logsize stays the same
          interval_start += interval_length;
          interval_state = 0;
        }
      }
      break;
    case 1:
      // we can't be at a leaf here: explore right subtree, except if it is off limits
      alt_start = interval_start + (1uL << (interval_logsize - 1));
      if(alt_start < num_elts) // alt_start is the right subtree start value
      {
        interval_logsize--;
        interval_start = alt_start;
        interval_state = 0;
      }
      else
      {
        // the whole right subinterval is off limits, declare the interval finished
        interval_state = 2;
      }

      break;
    case 2:
      //cout << "Finished processing subtree starting at " << interval_start << " and of size " << actual_length << endl;
      if(interval_logsize == logsize) return;
      else
      {
        // go up, adjusting interval start if on the right subtree
        if(is_right_subinterval)
        {
          interval_start -= interval_length; // alt start is the left subtree start
          interval_logsize++;
          interval_state = 2;
        }
        else
        {
          interval_logsize++;
          interval_state = 1;
        }
      }
      break;
    }
  }
}



