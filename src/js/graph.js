import {assert} from "./helper.js";
import {Mat} from "./mat.js";
import {sig} from "./math.js";
import _ from "lodash";

function processGate(target, name, descriptor) {
  let fn = descriptor.value;
  descriptor.value = function(...args) {
    let gate = fn.apply(this, args);
    gate.forward();
    this.needs_backprop && this.backprop.unshift(gate);
    return gate.props.out;
  }
}

// Transformer definitions
class Graph {
  constructor(needs_backprop=true) {
    
    this.needs_backprop = needs_backprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    this.backprop = [];
  }

  backward() {
    if(!this.needs_backprop)
      return;
    let b = this.backprop;

    for(let i=0,n=b.length;i<n;i++){
      b[i].backward();
    }
  }
  
  @processGate
  rowPluck(m, ix) {
    // pluck a row of m with index ix and return it as col vector
    return new RowPluckGate(m, ix);
  }

  @processGate
  tanh(m) {
    // tanh nonlinearity
    return new TanhGate(m);
  }

  @processGate
  sigmoid(m) {
    // sigmoid nonlinearity
    return new SigmoidGate(m);
  }

  @processGate
  relu(m) {
    return new ReluGate(m);
  }

  @processGate
  mul(m1, m2) {
    // multiply matrices m1 * m2
    return new MulGate(m1, m2);
  }

  @processGate
  add() {
    return new AddGate(...arguments);
  }

  @processGate
  eltmul(m1, m2) {
    return new EltmulGate(m1, m2);
  }
}

class EltmulGate {
  constructor(m1, m2) {
    this.props = {m1, m2};
  }

  forward() {
    let {m1, m2} = this.props;
    let out = new Mat(m1.n, m1.d);
    for(let i=0,n=m1.w.length;i<n;i++) {
      out.w[i] = m1.w[i] * m2.w[i];
    }
    this.props.out = out;
  }

  backward() {
    let {m1, m2, out} = this.props;

    for(let i=0,n=m1.w.length;i<n;i++) {
      m1.dw[i] += m2.w[i] * out.dw[i];
      m2.dw[i] += m1.w[i] * out.dw[i];
    }
  }
}
class AddGate {
  constructor() {
    this.props = {
      mats: arguments,
      m: arguments[0]
    };
  }

  forward() {
    let {mats, m} = this.props;

    let out = new Mat(m.n, m.d);
    for(let i=0,n=m.w.length;i<n;i++) {
      let sum = 0;
      for(let j=0,k=mats.length;j<k;j++){
        sum += mats[j].w[i];
      }
      out.w[i] = sum;
    }
    this.props.out = out;
  }
  
  backward() {
    let {mats, m, out} = this.props;

    for(let i=0,n=m.w.length;i<n;i++) {
      for(let j=0,k=mats.length;j<k;j++){
        mats[j].dw[i] += out.dw[i];
      }
    }
  }
}

class MulGate {
  constructor(m1, m2) {
    this.props = {m1, m2};
  }

  forward() {
    let {m1, m2} = this.props;
    let n = m1.n;
    let d = m2.d;
    let d1 = m1.d;

    let out = new Mat(n,d);

    for(let i=0;i<n;i++) { // loop over rows of m1
      for(let j=0;j<d;j++) { // loop over cols of m2
        let dot = 0.0;
        for(let k=0;k<d1;k++) { // dot product loop
          dot += m1.w[d1*i+k] * m2.w[d*k+j];
        }
        out.w[d*i+j] = dot;
      }
    }

    this.props.out = out;
  }

  backward() {
    let {m1, m2, out} = this.props;
    let n = m1.n;
    let d = m2.d;
    let d1 = m1.d;

    for(let i=0;i<n;i++) { // loop over rows of m1
      for(let j=0;j<d;j++) { // loop over cols of m2
        for(let k=0;k<d1;k++) { // dot product loop
          let b = out.dw[d*i+j];
          m1.dw[d1*i+k] += m2.w[d*k+j] * b;
          m2.dw[d*k+j] += m1.w[d1*i+k] * b;
        }
      }
    }
  }
}

class ReluGate {
  constructor(m) {
    this.props = {m};
  }

  forward() {
    let {m} = this.props;
    let out = new Mat(m.n, m.d);
    let n = m.w.length;
    for(let i=0;i<n;i++) { 
      out.w[i] = Math.max(0, m.w[i]); // relu
    }
    
    this.props.out = out;
  }

  backward() {
    let {m, out} = this.props;
    for(let i=0;i<n;i++) {
      m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
    }
  }
}

class SigmoidGate {
  constructor(m) {
    this.props = {m};
  }

  forward() {
    let {m} = this.props;  
    let out = new Mat(m.n, m.d);
    let n = m.w.length;
    for(let i=0;i<n;i++) { 
      out.w[i] = sig(m.w[i]);
    }
    this.props.out = out;
  }

  backward() {
    let {m, out} = this.props;
    let n = m.w.length;
    for(let i=0;i<n;i++) {
      // grad for z = tanh(x) is (1 - z^2)
      let mwi = out.w[i];
      m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
    }
  }
}

class TanhGate {
  constructor(m) {
    this.props = {m};
  }

  forward() {
    let {m} = this.props;
    let out = new Mat(m.n, m.d);
    let n = m.w.length;

    for(let i=0;i<n;i++) { 
      out.w[i] = Math.tanh(m.w[i]);
    }
    this.props.out = out;
  }

  backward() {
    let {m, out} = this.props;
    let n = m.w.length;

    for(let i=0;i<n;i++) {
      // grad for z = tanh(x) is (1 - z^2)
      let mwi = out.w[i];
      m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
    }
  }
}

class RowPluckGate {
  constructor(m, ix){
    this.props = {m, ix};
  }

  forward(){
    let {m, ix} = this.props;

    let d = m.d;
    let out = new Mat(d, 1);
    
    for(let i=0,n=d;i<n;i++){ 
      out.w[i] = m.w[d * ix + i];
    } // copy over the data
    
    this.props.out = out;
  }

  backward(){
    let {m, ix, out} = this.props;
    let d = m.d;

    for(let i=0,n=d;i<n;i++){
      m.dw[d * ix + i] += out.dw[i]; 
    }
  }
}

export default Graph;