import {assert} from "./helper.js";

// Random numbers utils

var randf = (a, b)=> Math.random()*(b-a)+a;
var randi = (a, b) => Math.floor(Math.random()*(b-a)+a);
var randn = (mu, std) => mu+gaussRandom()*std;


let gaussRandom = (() => {
  // polar form of the Box-Muller transformation
  // faster and more robust numerically
  // https://www.taygeta.com/random/gaussian.html

  let cache_value = null;

  return () => {
    if(cache_value!==null) {
      let temp = cache_value
      cache_value = null;
      return temp;
    }
    
    let u, v, r;

    while(!r || r >= 1){
      u = randf(-1, 1);
      v = randf(-1, 1);
      r = u*u + v*v;
    }

    let c = Math.sqrt(-2*Math.log(r)/r);
    cache_value = v*c;
    return u*c;
  }
})()

let fillRandn = (m, mu, std) => { 
  for(let i=0,n=m.w.length;i<n;i++) { 
    m.w[i] = randn(mu, std); 
  }
}

let fillRand = (m, lo, hi) => {
  for(let i=0,n=m.w.length;i<n;i++) {
    m.w[i] = randf(lo, hi); 
  }
}

let samplei = (w) => {
  // sample argmax from w, assuming w are 
  // probabilities that sum to one
  let r = randf(0,1);
  let x = 0.0;
  let i = 0;
  
  while(true) {
    x += w[i];
    if(x > r) { return i; }
    i++;
  }
  
  throw Error("Error sampling");
}

let maxi = (w) => {
  // argmax of array w
  let maxv = w[0];
  let maxix = 0;
  
  for(let i=1,n=w.length;i<n;i++) {
    let v = w[i];
    if(v > maxv) {
      maxix = i;
      maxv = v;
    }
  }
  return maxix;
}

// helper function returns array of zeros of length n
// and uses typed arrays if available
let zeros = (n) => {
  assert(!isNaN(n));

  if(typeof ArrayBuffer === 'undefined') {
    // lacking browser support
    let arr = new Array(n);
    for(let i=0;i<n;i++) { arr[i] = 0; }
    return arr;
  } else {
    return new Float64Array(n);
  }
}

let median = (values) => {
  values.sort((a,b) => a - b);
  let half = Math.floor(values.length/2);
  let ret;
  if(values.length % 2)
    ret = values[half];
  else
    ret = (values[half-1] + values[half]) / 2.0;
  return ret;
}

// helper function for computing sigmoid
let sig = (x) => 1.0/(1+Math.exp(-x));

export default {
	fillRandn,
	fillRand,
	randf,
	randi,
	randn,
	samplei,
	maxi,
  gaussRandom,
  zeros,
  median,
  sig
}