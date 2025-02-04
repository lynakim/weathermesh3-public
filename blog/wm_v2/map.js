const createDualMaps = () => {
  const animConfig = {
    stepsRight: 25,
    stepsDown: 10,
    stepInterval: 200,
    gridSize: 2,
    boxSize: 7,
    smallBoxSize: 1,
    startLon: -120,
    startLat: 50,
  };

  let currentX = 0;
  let currentY = 0;
  let animationTimer;

  const createMap = (containerId, title) => {
    const width = 350;
    const height = 300;
    const res = animConfig.gridSize;

    const svg = d3.select(containerId)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);

    // Add white background and title text


    const projection = d3.geoEquirectangular()
      .center([-90.5, 33.5])
      .scale(300)
      .translate([width / 2, height / 2]);
    const path = d3.geoPath().projection(projection);

    const createBoxGeometry = (lon, lat, size) => {
      const coords = [];
      for (let x = lon; x <= lon + size; x += 0.1) {
          coords.push([x, lat]);
      }
      for (let y = lat; y >= lat - size; y -= 0.1) {
          coords.push([lon + size, y]);
      }
      for (let x = lon + size; x >= lon; x -= 0.1) {
          coords.push([x, lat - size]);
      }
      for (let y = lat - size; y <= lat; y += 0.1) {
          coords.push([lon, y]);
      }
      coords.push([lon, lat]);
      return coords;
    };

    const gridGroup = svg.append('g').attr('class', 'grid-lines');
    const graticule = d3.geoGraticule()
      .step([res, res])
      .extent([[-140, 0], [-20, 60]]);

    gridGroup.append('path')
      .datum(graticule)
      .attr('d', path)
      .attr('stroke', 'gray')
      .attr('fill', 'none')
      .attr('stroke-width', 0.5);

    const largeBoxGroup = svg.append('g');
    const smallBoxGroup = svg.append('g');


    svg.append('rect')
    .attr('x', 15)
    .attr('y', 15)
    .attr('width', 75)
    .attr('height', 25)
    .attr('fill', 'white');

    svg.append('text')
    .attr('x', 20)
    .attr('y', 34)
    .attr('font-size', '16px')
    .style('filter', 'drop-shadow(0 0 5px white)')
    .text(title);

    return {
      svg,
      projection,
      path,
      updateBoxes: (lon1, lat1, lon2, lat2) => {
        const largeBox = {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: createBoxGeometry(lon1, lat1, animConfig.boxSize * res)
          }
        };

        largeBoxGroup.selectAll('path').remove();
        largeBoxGroup.append('path')
          .datum(largeBox)
          .attr('d', path)
          .attr('stroke', 'red')
          .attr('fill', 'none')
          .attr('stroke-width', 2);

        const smallBox = {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [createBoxGeometry(
              lon2 + 3 * res, 
              lat2 - 3 * res, 
              animConfig.smallBoxSize * res
            )]
          }
        };

        smallBoxGroup.selectAll('path').remove();
        smallBoxGroup.append('path')
          .datum(smallBox)
          .attr('d', path)
          .attr('fill', 'red')
          .attr('fill-opacity', 0.5);
      }
    };
  };

  // Create two map instances with titles
  const map1 = createMap('#map-container-1', 'NATTEN');
  const map2 = createMap('#map-container-2', 'SWIN');

  // Create single shared legend
  const legendSvg = d3.select('#map-container-1')
    .append('svg')
    .attr('width', 300)
    .attr('height', 80)
    .attr('style', 'display: block; margin-top: -10px;');

  const legend = legendSvg.append('g')
    .attr('transform', 'translate(20, 20)');

  // Legend box for attention window
  legend.append('rect')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', 20)
    .attr('height', 20)
    .attr('stroke', 'red')
    .attr('fill', 'none')
    .attr('stroke-width', 2);

  legend.append('text')
    .attr('x', 30)
    .attr('y', 15)
    .text('Attention Window');

  // Legend box for patch
  legend.append('rect')
    .attr('x', 0)
    .attr('y', 30)
    .attr('width', 20)
    .attr('height', 20)
    .attr('fill', 'red')
    .attr('fill-opacity', 0.5);

  legend.append('text')
    .attr('x', 30)
    .attr('y', 45)
    .text('Updated Patch');

  // Load coastlines and start animation
  d3.json('land-110m.json').then(function(worldData) {
    const landFeature = topojson.feature(worldData, worldData.objects.land);
    
    map1.svg.insert('path', ':first-child')
    .datum(landFeature)
    .attr('d', map1.path)
    .attr('fill', 'none')
    .attr('stroke', 'black')
    .attr('stroke-width', 0.5);
  
  map2.svg.insert('path', ':first-child')
    .datum(landFeature)
    .attr('d', map2.path)
    .attr('fill', 'none')
    .attr('stroke', 'black')
    .attr('stroke-width', 0.5);
  

    map1.updateBoxes(animConfig.startLon, animConfig.startLat, animConfig.startLon, animConfig.startLat);
    map2.updateBoxes(animConfig.startLon, animConfig.startLat, animConfig.startLon, animConfig.startLat);

    const animate = () => {
      const lon = animConfig.startLon + (currentX * animConfig.gridSize);
      const lat = animConfig.startLat - (currentY * animConfig.gridSize);
      
      map1.updateBoxes(lon, lat, lon, lat);
      
      const res = animConfig.gridSize;
      const bb = animConfig.boxSize;
      map2.updateBoxes(bb-1+Math.floor(lon / (bb*res))*(bb*res), bb-1+Math.floor(lat / (bb*res))*(bb*res), lon, lat);

      currentX++;
      if (currentX >= animConfig.stepsRight) {
        currentX = 0;
        currentY++;
        if (currentY >= animConfig.stepsDown) {
          currentY = 0;
        }
      }
    };

    animationTimer = setInterval(animate, animConfig.stepInterval);
  });

  return {
    map1,
    map2,
    stop: () => clearInterval(animationTimer),
    start: () => {
      clearInterval(animationTimer);
      currentX = 0;
      currentY = 0;
      animationTimer = setInterval(animate, animConfig.stepInterval);
    },
    config: animConfig
  };
};