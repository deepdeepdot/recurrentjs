import {benchmark} from "./helper";

class Ticker {
	constructor() {
		this.timer = null;
		this.tick_iter = 0;
		this.schedule = {};
	}

	every_tick(fnc) {
		this.fnc = fnc;
	}

	every(ms, fnc){
		this.schedule[ms] = this.schedule[ms] || [];
		this.schedule[ms].push(fnc.bind(this));
	}

	start() {
		this.stop();
		
		let tick = () => {
			this.timer = setTimeout(() => {
				this.tick_iter += 1;
				
				this.tick_time = benchmark(this.fnc);

				for(let ms in this.schedule){
					if (this.tick_iter % ms !== 0)
						continue;
					
					for(let i in this.schedule[ms]){
						this.schedule[ms][i]();
					}
				}

				tick();
			}, 0);
		}

		tick();
	}

	reset() {
		this.stop();
		this.tick_iter = 0;
	}

	stop() {
		if(!this.timer)
			return;
		
		clearTimeout(this.timer);
		this.timer = null;
	}
}

export default Ticker;