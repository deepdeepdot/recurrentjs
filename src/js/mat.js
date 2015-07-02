import {assert} from "./helper.js";
import {fillRandn, fillRand, zeros} from "./math.js";

// Mat holds a matrix
class Mat {
  constructor(n, d) {
    // n is number of rows d is number of columns
    this.reset(n, d);
  }

  get(row, col) { 
    // slow but careful accessor function
    // we want row-major order
    var ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  }

  set(row, col, v) {
    // slow but careful accessor function
    var ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v; 
  }
  
  reset(n, d) {
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  }

  toJSON() {
    var json = {
      'n': this.n,
      'd': this.d,
      'w': this.w
    };
    return json;
  }
  
  fromJSON(json) {
    let {n, d} = json
    this.reset(n, d);

    for(let i=0, t=(n * d); i<t; i++) {
      this.w[i] = json.w[i]; // copy over weights
    }
  }
}

// return Mat but filled with random numbers from gaussian
var RandMat = (n,d,mu,std) => {
  var m = new Mat(n, d);
  fillRand(m, mu, std);
  return m;
}

// Mat utils
// fill matrix with random gaussian numbers

var softmax = (m) => {
  var out = new Mat(m.n, m.d); // probability volume
  var maxval = -999999;
  
  for(var i=0,n=m.w.length;i<n;i++) { 
    if(m.w[i] > maxval) maxval = m.w[i]; 
  }

  var s = 0.0;
  
  for(var i=0,n=m.w.length;i<n;i++) { 
    out.w[i] = Math.exp(m.w[i] - maxval);
    s += out.w[i];
  }
  
  for(var i=0,n=m.w.length;i<n;i++) { 
    out.w[i] /= s; 
  }

  // no backward pass here needed
  // since we will use the computed probabilities outside
  // to set gradients directly on m
  return out;
}

export default {
  RandMat,
  Mat,
  softmax
}