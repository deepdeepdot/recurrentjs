import {forwardIndex, forwardLSTM, initLSTM, forwardRNN, initRNN} from "../recurrent";
import {RandMat, Mat, softmax} from "../mat.js";
import {median, randi, maxi, samplei, gaussRandom} from "../math";
import Graph from "../graph.js";

var max_chars_gen = 100; // max length of generated sentences

var predictSentence = (model, samplei_bool=false, temperature=1.0, letterToIndex, indexToLetter, logprobs, generator, hidden_sizes) => {
  var G = new Graph(false);
  var s = '';
  var prev = {};
  while(true) {

    // RNN tick
    var ix = s.length === 0 ? 0 : letterToIndex[s[s.length-1]];
    var lh = forwardIndex(G, model, ix, prev, generator, hidden_sizes);
    prev = lh;

    // sample predicted letter
    logprobs = lh.o;
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
    let ix;

    if(samplei_bool) {
      ix = samplei(probs.w);
    } else {
      ix = maxi(probs.w);  
    }
    
    if(ix === 0) break; // END token predicted, break out
    if(s.length > max_chars_gen) break; // something is wrong

    var letter = indexToLetter[ix];
    s += letter;
  }
  return s;
}

module.exports = function (self) {
  self.addEventListener('message',function (ev){
  	let [random_id, num_lines, args] = ev.data;
  	let lines = [];
  	
    for(let i = 0; i<num_lines; i++){
  		lines.push(predictSentence(...args));
  	}

    let sentences = lines.join("\n");
  	self.postMessage([random_id, sentences]);
  });
};