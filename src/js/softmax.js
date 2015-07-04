import {Mat} from "./mat";

let softmax = (m) => {
  let out = new Mat(m.n, m.d); // probability volume
  let maxval = -999999;
  
  for(let i=0,n=m.w.length;i<n;i++) { 
    if(m.w[i] > maxval) maxval = m.w[i]; 
  }

  let s = 0.0;
  
  for(let i=0,n=m.w.length;i<n;i++) { 
    out.w[i] = Math.exp(m.w[i] - maxval);
    s += out.w[i];
  }
  
  for(let i=0,n=m.w.length;i<n;i++) { 
    out.w[i] /= s; 
  }

  // no backward pass here needed
  // since we will use the computed probabilities outside
  // to set gradients directly on m
  return out;
}

export default softmax;