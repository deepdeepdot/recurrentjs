import {assert} from "./helper.js";
import {randn, zeros} from "./math.js";

// Mat holds a matrix
class Mat {
  constructor(n, d, opt={}) {
    // n is number of rows d is number of columns
    this.reset(n, d, opt);
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
  
  reset(n, d, opt) {
    this.n = n;
    this.d = d;
    this.opt = opt;
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
class RandMat extends Mat{
  constructor(n, d, opt){
    super(n, d, opt);
  }

  reset(n, d, opt){
    super(n, d, opt);
    let mu = opt.mu || 0;
    let std = opt.std || 0.08;
    let w = this.w;

    for(let i=0,t=(n*d);i<t;i++) { 
      w[i] = randn(mu, std); 
    }
  }
}

export default {
  RandMat,
  Mat
}