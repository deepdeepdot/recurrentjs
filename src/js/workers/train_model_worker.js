import Ticker from "../ticker";
import _ from "lodash";
import {costfun} from "../recurrent";
import Solver from "../solver"; 

let ticker = new Ticker();
let solver = new Solver();

let cost_struct, solver_stats;
let data_sents, generator, letterToIndex, hidden_sizes, model, learning_rate, regc, clipval, epoch_size;
let ppl_list = [];

ticker.every_tick(function(){
  // sample sentence from data
  let sent = _.sample(data_sents)

  // evaluate cost function on a sentence
  cost_struct = costfun(model, sent, letterToIndex, generator, hidden_sizes);
  
  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();
  
  // perform param update
  solver_stats = solver.step(model, learning_rate, regc, clipval);
  
  ppl_list.push(cost_struct.ppl); // keep track of perplexity
  // evaluate now and then
})

module.exports = function (self) {
  self.addEventListener('message',function (ev){
    let [id, cmd, data] = ev.data;
    
    switch(cmd){
      case 'init':
        cost_struct = null;
        ppl_list = [];
        ticker.reset();

        data_sents =data.data_sents;
        generator = data.generator;
        letterToIndex = data.letterToIndex;
        hidden_sizes = data.hidden_sizes;
        model = data.model;
        learning_rate = data.learning_rate;
        regc = data.regc;
        clipval = data.clipval;
        self.postMessage([id, "init"]);
        break;
      case 'start':
        ticker.start();
        self.postMessage([id, "start"]);
        break;
      case 'stop':
        ticker.stop();
        self.postMessage([id, "stop"]);
        break;
      case 'reset':
        cost_struct = null;
        ppl_list = [];
        ticker.reset();
        self.postMessage([id, "reset"]);
        break;
      case 'update_learning_rate':
        learning_rate = data.learning_rate
      case 'sample':
        if(!cost_struct){ return; }
        self.postMessage([id, {
          model: model,
          tick_iter: ticker.tick_iter,
          tick_time: ticker.tick_time,
          ppl: cost_struct.ppl
        }]);
        break;
      case 'sample_ppl':
        self.postMessage([id, {
          ppl_list,
          tick_iter: ticker.tick_iter
        }]);
        ppl_list = [];
        break;
    }
  });
};