class WebWorker {
  constructor(worker){
    this.worker = worker;
    this.promises = {};

    let p = this.promises;

    this.worker.addEventListener('message', function(ev) {
      let [id, data] = ev.data;

      if(p[id]){
        p[id].resolve(data);
        delete p[id];
      }
    });
  }

  send_work(args, cb){
    let id = Math.floor(Math.random()*1000000);
    args.unshift(id);

    let p = Promise.defer();
    this.promises[id] = p;
    this.worker.postMessage(args);
    return p.promise;
  }
}

export default WebWorker