import Ticker from "../ticker";
import {costfun} from "../recurrent";
import Solver from "../solver";
import {randi} from "../math"; 

let ticker = new Ticker();
let solver = new Solver();

let cost_struct, solver_stats;
let input_text, input_text_length, generator, letterToIndex, hidden_sizes, model, learning_rate, regc, clipval, epoch_size, train_seq_length;
let ppl_list = [];

ticker.every_tick(function(){
  // sample sentence from data
  let randind = randi(0, input_text_length - train_seq_length);
  let sent = input_text.substr(randind, train_seq_length);

  cost_struct = costfun(model, sent, letterToIndex, generator, hidden_sizes);
  
  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();

  // perform param update
  solver_stats = solver.step(model, learning_rate, regc, clipval);
  
  ppl_list.push(cost_struct.ppl); // keep track of perplexity
})

module.exports = function (self) {
  self.addEventListener('message',function (ev){
    let [id, cmd, data] = ev.data;
    
    switch(cmd){
      case 'init':
        ticker.reset();

        cost_struct = null;
        ppl_list = [];
        input_text =data.input_text;
        input_text_length = data.input_text.length;
        generator = data.generator;
        letterToIndex = data.letterToIndex;
        hidden_sizes = data.hidden_sizes;
        model = data.model;
        learning_rate = data.learning_rate;
        train_seq_length = data.train_seq_length;
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
        if(!cost_struct || !cost_struct.ppl){ return; }
        self.postMessage([id, {
          tick_iter: ticker.tick_iter,
          tick_time: ticker.tick_time,
          ppl: cost_struct.ppl
        }]);
        break;
      case 'sample_model':
        self.postMessage([id, {model}]);
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