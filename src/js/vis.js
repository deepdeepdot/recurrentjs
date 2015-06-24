import d3 from "d3-browserify";
// can be used to graph loss, or accuract over time

class Graph {
  
  constructor(selector, options={}){
    this.graph = d3.select(selector);
    this.reset();
  }

  reset() {
    this.pts && this.pts.length && this.clearGraph();
    
    this.pts = [];
    this.maxy = -9999;
    this.miny = 9999;
  }

  clearGraph() {
    this.graph.selectAll("*").remove();
  }

  add(step, y) {
    var time = +new Date(); // in ms
    if(y>this.maxy*0.99) this.maxy = y*1.05;
    if(y<this.miny*1.01) this.miny = y*0.95;

    this.pts.push({step, time, y});
  }
  
  drawSelf(canv) {
    if(this.pts.length < 2)
      return;

    var m = {top: 10, right: 10, bottom: 20, left: 20};
    var w = 250 - m.right - m.left;
    var h = 200 - m.top - m.bottom;
  
    let first_pt = this.pts[0];
    let last_pt = this.pts[this.pts.length - 1];

    var x = d3.scale.linear().domain([first_pt.step, last_pt.step]).range([0, w]);
    var y = d3.scale.linear().domain([this.miny, this.maxy]).range([h, 0]);
      
    var line = d3.svg.line()
      .x((d) => x(d.step))
      .y((d) => y(d.y));
    
    this.clearGraph();

    let graph = this.graph.append("svg:svg")
      .attr("width", w + m.left + m.right)
      .attr("height", h + m.bottom + m.top)
      .append("svg:g")
      .attr("transform", `translate(${m.left}, ${m.top})`);

    var xAxis = d3.svg.axis()
      .scale(x)
      .innerTickSize(-h)
      .outerTickSize(0)
      .ticks(6)
      .tickFormat((d) => `${d/1000}k` )
      .orient("bottom");

    graph.append("svg:g")
      .attr("class", "x axis")
      .attr("transform", `translate(0, ${h})`)
      .call(xAxis);

    var yAxis = d3.svg.axis()
      .scale(y)
      .innerTickSize(-w)
      .outerTickSize(0)
      .ticks(6)
      .orient("left");

    graph.append("svg:g")
      .attr("class", "y axis")
      .attr("transform", `translate(0, 0)`)
      .call(yAxis);
    
    graph.selectAll('circle')
      .data(this.pts)
      .enter().append('circle')
      .attr('cx', (d)=> x(d.step))
      .attr('cy', (d)=> y(d.y))
      .attr('r', 3); 
    
    graph.append("svg:path")
      .attr("d", line(this.pts));
  }
}

export default {
  Graph
};