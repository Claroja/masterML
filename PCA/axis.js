function axix(svg,xdomain,ydomain){
	width = svg.style()[0][0].width["baseVal"]["value"]  //获得svg的宽度
	height = svg.style()[0][0].height["baseVal"]["value"]  //获得svg的高度
	x_zero = (0-d3.min(xdomain))/(d3.max(xdomain)-d3.min(xdomain))  //标准化坐标系，用于确定0点的位置
	y_zero = (0-d3.min(ydomain))/(d3.max(ydomain)-d3.min(ydomain))  //标准化坐标系，用于确定0点的位置
	
	var xlinear = d3.scale.linear()
		.domain([xdomain[0], xdomain[1]])
		.range([0, width]);  //结合transform,x轴坐标向右平移10px
	var ylinear = d3.scale.linear()
		.domain([ydomain[0], ydomain[1]])
		.range([height, 0]);  //结合transform,x轴坐标向右平移10px
	var xaxis = d3.svg.axis()
		.scale(xlinear)      //指定比例尺
		.orient("bottom")   //指定刻度的方向
		.ticks(10);          //指定刻度的数量
	var yaxis = d3.svg.axis()
		.scale(ylinear)
		.orient("left")
		.ticks(10);
	svg.append("g")
		.attr("class","axis")
		.attr("transform","translate(0,"+(height)*(1-y_zero)+")")  //x轴上下平移不会对坐标系造成影响
		.call(xaxis);
	svg.append("g")
		.attr("class","axis")
		.attr("transform","translate("+(width)*x_zero+",0)")  //y轴左右平移不会对坐标系造成影响
		.call(yaxis);
		
	svg.selectAll('.axis text[y="9"]')  //将x轴的0标左移动
	.each(function (d, i) {
		if ( d == 0 ) {
		this.remove()
		}});
	svg.selectAll('.axis text[x="-9"]') //删除y轴0标
	.each(function (d, i) {
		if ( d == 0 ) {
		this.remove()
		}});
// 	svg.selectAll('.axis text')  //删除-1标
// 	.each(function (d, i) {
// 		if ( d == -1 ) {
// 		this.remove()
// 		}});
	return [xlinear,ylinear]
	}