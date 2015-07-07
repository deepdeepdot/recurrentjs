import {forwardIndex, forwardLSTM, initLSTM, forwardRNN, initRNN} from "../recurrent";
import softmax from "../softmax"
import {RandMat, Mat} from "../mat";
import {median, randi, maxi, samplei, gaussRandom} from "../math";
import Graph from "../graph";
import _ from "lodash";

var max_chars_gen = 100; // max length of generated sentences

var predictSentence = (num_characters, model, samplei_bool=false, temperature=1.0, letterToIndex, indexToLetter, generator, hidden_sizes) => {
  let G = new Graph(false);
  let sentences = [];

  let s = '';
  let ix;
  let prev = {};
  
  while(true) {

    // RNN tick
    let ix;

    if(s.length === 0){
      ix = 0;
    } else {
      ix = letterToIndex[_.last(s)];
    }

    var lh = forwardIndex(G, model, ix, prev, generator, hidden_sizes);
    prev = lh;

    // sample predicted letter
    let logprobs = lh.o;
    if(temperature !== 1.0 && samplei_bool) {
      // scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky
      for(var q=0,nq=logprobs.w.length;q<nq;q++) {
        logprobs.w[q] /= temperature;
      }
    }

    let probs = softmax(logprobs);

    if(samplei_bool) {
      ix = samplei(probs.w);
    } else {
      ix = maxi(probs.w);  
    }
    
    if(ix === 0) break; // END token predicted, break out
    if(s.length > num_characters) break; // something is wrong

    let letter = indexToLetter[ix];
    s += letter;
  }
  return s;
}

module.exports = function (self) {
  self.addEventListener('message',function (ev){
  	let [id, ...args] = ev.data;
		let sample = predictSentence(...args);
  	self.postMessage([id, sample]);
  });
};