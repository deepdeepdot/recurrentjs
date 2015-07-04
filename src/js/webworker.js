class WebWorker {
  constructor(worker){
    this.worker = worker;
    this.callbacks = {};

    let c = this.callbacks;

    this.worker.addEventListener('message', function(ev) {
      let [id, data] = ev.data;

      if(c[id]){
        c[id](data);
        delete c[id];
      }
    });
  }

  send_work(args, cb){
    let id = Math.floor(Math.random()*1000000);
    args.unshift(id);
    this.callbacks[id] = cb;
    this.worker.postMessage(args);
  }
}

export default WebWorker