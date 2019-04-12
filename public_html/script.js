      google.charts.load('current', {
        'packages':['geochart'],
        // Note: you will need to get a mapsApiKey for your project.
        // See: https://developers.google.com/chart/interactive/docs/basic_load_libs#load-settings
        'mapsApiKey': 'AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY'
      });
      google.charts.setOnLoadCallback(drawRegionsMap);

      function drawRegionsMap() {
       var data = new google.visualization.DataTable();
       data.addColumn('string','Country');
       data.addColumn('number', 'Cluster');
       for (var i = 0; i < data1.length; i++) {
                data.addRows(1);
                country = data1[i]["Country"]
                cluster = data1[i]["Cluster"]
                data.setValue(i, 0, country);
                data.setValue(i, 1, cluster)
        };

        var options = {colorAxis: {
          colors: ['red', 'green', 'purple', 'blue', 'yellow', 'orange'],
          values: [0, 1, 2, 3, 4, 5]
        }};

        var chart = new google.visualization.GeoChart(document.getElementById('regions_div'));

        chart.draw(data, options);
      }
