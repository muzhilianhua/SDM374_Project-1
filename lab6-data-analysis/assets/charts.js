(function () {
  var D = window.REPORT_DATA;
  if (!D) return;
  var s = getComputedStyle(document.documentElement);
  var accent = s.getPropertyValue('--accent').trim();
  var accent2 = s.getPropertyValue('--accent2').trim();
  var ink = s.getPropertyValue('--ink').trim();
  var muted = s.getPropertyValue('--muted').trim();
  var rule = s.getPropertyValue('--rule').trim();
  var font = "'WorkSans','Microsoft YaHei','PingFang SC',sans-serif";

  var tips = {
    textStyle: { color: ink, fontFamily: font, fontSize: 12 },
    borderColor: rule,
    backgroundColor: 'rgba(255,255,255,0.96)',
    extraCssText: 'box-shadow:0 4px 16px rgba(21,33,29,0.12);'
  };
  function ax(extra) {
    var o = {
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: muted, fontFamily: font, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      nameTextStyle: { color: muted, fontFamily: font, fontSize: 11 }
    };
    for (var k in extra) o[k] = extra[k];
    return o;
  }
  function init(id) {
    var el = document.getElementById(id);
    if (!el) return null;
    var c = echarts.init(el, null, { renderer: 'svg' });
    window.addEventListener('resize', function () { c.resize(); });
    return c;
  }
  var legendStyle = { color: muted, fontFamily: font, fontSize: 11, itemWidth: 12, itemHeight: 8 };

  /* ================= tables ================= */
  var tb = document.getElementById('tbl-weight-stats');
  if (tb && D.weights && D.weights.stats) {
    tb.innerHTML = D.weights.stats.map(function (r) {
      return '<tr><td>' + r.metric + '</td>' +
        '<td class="num">' + r.mean.toFixed(4) + '</td>' +
        '<td class="num">' + r.std.toFixed(4) + '</td>' +
        '<td class="num">' + r.min.toFixed(4) + '</td>' +
        '<td class="num">' + r.q05.toFixed(4) + '</td>' +
        '<td class="num">' + r.median.toFixed(4) + '</td>' +
        '<td class="num">' + r.q95.toFixed(4) + '</td>' +
        '<td class="num">' + r.max.toFixed(4) + '</td></tr>';
    }).join('');
  }
  var tt = document.getElementById('tbl-top');
  if (tt && D.top && D.top.table) {
    var cols = ['b1_ramping', 'b2_ramping', 'b3_ramping', 'b1_cost', 'b1_carbon', 'b1_degradation', 'b1_peak'];
    tt.innerHTML = D.top.table.map(function (r) {
      return '<tr><td>' + r.id + '</td><td class="num">' + r.target.toFixed(2) + '</td>' +
        cols.map(function (k) { return '<td class="num">' + r[k].toFixed(3) + '</td>'; }).join('') + '</tr>';
    }).join('');
  }

  /* ================= 1. ramping distribution ================= */
  var c1 = init('chart-ramping');
  if (c1 && D.ramping) {
    var bw = D.ramping.hist.centers[1] - D.ramping.hist.centers[0];
    var dens = D.ramping.kde.y.map(function (v) { return v * D.ramping.values.length * bw; });
    c1.setOption({
      animation: false,
      grid: { left: 54, right: 56, top: 30, bottom: 50 },
      tooltip: Object.assign({ trigger: 'axis' }, tips),
      legend: { data: ['样本数', '核密度'], right: 8, top: 0, textStyle: legendStyle },
      xAxis: ax({
        type: 'category',
        data: D.ramping.hist.centers.map(function (v) { return v.toFixed(0); }),
        name: 'ramping',
        boundaryGap: true,
        axisLabel: { color: muted, fontFamily: font, fontSize: 10, interval: 3 }
      }),
      yAxis: [
        ax({ type: 'value', name: '样本数' }),
        ax({ type: 'value', name: '密度', splitLine: { show: false } })
      ],
      series: [
        {
          name: '样本数', type: 'bar', data: D.ramping.hist.counts, barWidth: '86%',
          itemStyle: { color: accent + 'aa' },
          markLine: {
            symbol: 'none', silent: true,
            lineStyle: { color: accent2, type: 'dashed', width: 1 },
            label: { color: accent2, fontFamily: font, fontSize: 10, formatter: '均值 160.45' },
            data: [{ yAxis: D.ramping.stats.mean }]
          }
        },
        {
          name: '核密度', type: 'line', yAxisIndex: 1, data: dens, smooth: true, symbol: 'none',
          lineStyle: { color: accent2, width: 1.6 }
        }
      ]
    });
  }

  /* ================= 2. weight boxplot ================= */
  var c2 = init('chart-box');
  if (c2 && D.weights && D.weights.box) {
    var boxColors = [accent, accent2, '#6f8079'];
    var cats = [], bdata = [];
    D.weights.metrics.forEach(function (m, mi) {
      for (var b = 0; b < 3; b++) {
        cats.push(m + '·' + (b + 1));
        bdata.push({
          value: D.weights.box[mi][b],
          itemStyle: { color: boxColors[b] + '2e', borderColor: boxColors[b], borderWidth: 1.3 }
        });
      }
    });
    function legendItem(right, color, text) {
      return {
        type: 'group', right: right, top: 2,
        children: [
          { type: 'rect', shape: { x: 0, y: 1, width: 9, height: 9 }, style: { fill: color + '2e', stroke: color, lineWidth: 1 } },
          { type: 'text', style: { x: 13, y: 0, text: text, fill: muted, fontFamily: font, fontSize: 11 } }
        ]
      };
    }
    c2.setOption({
      animation: false,
      grid: { left: 52, right: 20, top: 34, bottom: 58 },
      tooltip: Object.assign({ trigger: 'item' }, tips),
      graphic: [
        legendItem(118, boxColors[2], '3 号楼'),
        legendItem(70, boxColors[1], '2 号楼'),
        legendItem(22, boxColors[0], '1 号楼')
      ],
      xAxis: ax({
        type: 'category', data: cats, boundaryGap: true,
        axisLabel: { color: muted, fontFamily: font, fontSize: 9, rotate: 48, interval: 0 }
      }),
      yAxis: ax({ type: 'value', name: '权重', max: 1 }),
      series: [{
        name: '权重分布', type: 'boxplot', data: bdata, boxWidth: 12
      }]
    });
  }

  /* ================= 3/4. CLR & ALR rho ================= */
  function rhoBar(id, rows) {
    var c = init(id);
    if (!c || !rows) return;
    var top = rows.slice(0, 10).slice().reverse();
    var lo = Math.min.apply(null, top.map(function (r) { return r.rho; }));
    var hi = Math.max.apply(null, top.map(function (r) { return r.rho; }));
    c.setOption({
      animation: false,
      grid: { left: 118, right: 52, top: 10, bottom: 34 },
      tooltip: Object.assign({
        trigger: 'axis', axisPointer: { type: 'shadow' },
        formatter: function (ps) {
          var p = ps[0];
          return p.name + '<br/>Spearman ρ = ' + p.value.toFixed(3);
        }
      }, tips),
      xAxis: ax({ type: 'value', min: Math.min(-0.1, lo - 0.06), max: hi + 0.08, name: 'Spearman ρ' }),
      yAxis: ax({ type: 'category', data: top.map(function (r) { return r.feature; }) }),
      series: [{
        type: 'bar', barWidth: '58%',
        data: top.map(function (r) {
          return { value: r.rho, itemStyle: { color: r.rho >= 0 ? accent2 : accent } };
        }),
        label: {
          show: true, color: muted, fontFamily: font, fontSize: 10,
          position: 'right',
          formatter: function (p) { return p.value.toFixed(2); }
        }
      }]
    });
  }
  if (D.coda) {
    rhoBar('chart-clr-rho', D.coda.clr.rho);
    rhoBar('chart-alr-rho', D.coda.alr.rho);
  }

  /* ================= 5. raw Spearman matrix ================= */
  var c3 = init('chart-corr');
  if (c3 && D.corr) {
    var L = D.corr.labels, M = D.corr.matrix, cells = [];
    for (var i = 0; i < L.length; i++) for (var j = 0; j < L.length; j++) cells.push([j, i, M[i][j]]);
    c3.setOption({
      animation: false,
      grid: { left: 96, right: 28, top: 22, bottom: 104 },
      tooltip: Object.assign({
        formatter: function (p) { return L[p.value[1]] + ' × ' + L[p.value[0]] + '<br/>Spearman ρ = ' + p.value[2].toFixed(3); }
      }, tips),
      xAxis: ax({
        type: 'category', data: L, splitArea: { show: false },
        axisLabel: { color: muted, fontFamily: font, fontSize: 9, rotate: 55, interval: 0 }
      }),
      yAxis: ax({
        type: 'category', data: L, splitArea: { show: false },
        axisLabel: { color: muted, fontFamily: font, fontSize: 9, interval: 0 }
      }),
      visualMap: {
        min: -1, max: 1, calculable: false, orient: 'horizontal', left: 'center', bottom: 6,
        itemWidth: 12, itemHeight: 110, textStyle: { color: muted, fontFamily: font, fontSize: 10 },
        inRange: { color: [accent2, '#f7f9f8', accent] }
      },
      series: [{
        type: 'heatmap', data: cells, splitArea: { show: false },
        label: {
          show: true, color: ink, fontFamily: font, fontSize: 7,
          formatter: function (p) { return Math.abs(p.value[2]) >= 0.30 ? p.value[2].toFixed(2) : ''; }
        },
        emphasis: { itemStyle: { borderColor: ink, borderWidth: 1 } }
      }]
    });
  }

  /* ================= 6. top vs bottom ================= */
  var c5 = init('chart-top');
  if (c5 && D.top && D.corr) {
    c5.setOption({
      animation: false,
      grid: { left: 52, right: 20, top: 34, bottom: 96 },
      tooltip: Object.assign({ trigger: 'axis', axisPointer: { type: 'shadow' } }, tips),
      legend: { data: ['最优 5%', '最差 5%'], top: 0, right: 6, textStyle: legendStyle },
      xAxis: ax({
        type: 'category', data: D.corr.labels.slice(0, 21),
        axisLabel: { color: muted, fontFamily: font, fontSize: 9, rotate: 55, interval: 0 }
      }),
      yAxis: ax({ type: 'value', name: '平均权重', max: 0.35 }),
      series: [
        { name: '最优 5%', type: 'bar', data: D.top.top_mean, itemStyle: { color: accent } },
        { name: '最差 5%', type: 'bar', data: D.top.bottom_mean, itemStyle: { color: accent2 + 'cc' } }
      ]
    });
  }

  /* ================= 7. Cohen's d ================= */
  var c6 = init('chart-effect');
  if (c6 && D.top && D.top.effects) {
    var ef = D.top.effects.slice().reverse();
    c6.setOption({
      animation: false,
      grid: { left: 118, right: 50, top: 12, bottom: 34 },
      tooltip: Object.assign({
        trigger: 'axis', axisPointer: { type: 'shadow' },
        formatter: function (ps) {
          var p = ps[0], row = null;
          for (var i = 0; i < D.top.effects.length; i++) if (D.top.effects[i].feature === p.name) row = D.top.effects[i];
          if (!row) return p.name + '<br/>d = ' + p.value.toFixed(3);
          return p.name + '<br/>最优 ' + row.top.toFixed(3) + ' · 最差 ' + row.bottom.toFixed(3) + '<br/>Cohen\'s d = ' + row.d.toFixed(2);
        }
      }, tips),
      xAxis: ax({ type: 'value', name: "Cohen's d" }),
      yAxis: ax({ type: 'category', data: ef.map(function (r) { return r.feature; }) }),
      series: [{
        type: 'bar', barWidth: '56%',
        data: ef.map(function (r) { return { value: r.d, itemStyle: { color: r.d >= 0 ? accent : accent2 } }; }),
        label: { show: true, position: 'right', color: muted, fontFamily: font, fontSize: 10, formatter: function (p) { return p.value.toFixed(2); } }
      }]
    });
  }

  /* ================= 8/9. PCA scatter ================= */
  function pcaScatter(id, pts, xname, yname) {
    var c = init(id);
    if (!c || !pts) return;
    c.setOption({
      animation: false,
      grid: { left: 56, right: 18, top: 18, bottom: 48 },
      tooltip: Object.assign({
        trigger: 'item',
        formatter: function (p) {
          return xname + ' = ' + p.value[0].toFixed(2) + '<br/>' + yname + ' = ' + p.value[1].toFixed(2) + '<br/>ramping = ' + p.value[2].toFixed(1);
        }
      }, tips),
      xAxis: ax({ type: 'value', name: xname, scale: true, splitLine: { show: false } }),
      yAxis: ax({ type: 'value', name: yname, scale: true }),
      visualMap: {
        min: 116, max: 243, dimension: 2, calculable: false, orient: 'horizontal',
        left: 'center', bottom: 2, itemWidth: 110, itemHeight: 8,
        textStyle: { color: muted, fontFamily: font, fontSize: 10 },
        inRange: { color: [accent, '#e8efec', accent2] }
      },
      series: [{
        type: 'scatter', data: pts, symbolSize: 4.5, large: true,
        itemStyle: { opacity: 0.6 }
      }]
    });
  }
  if (D.coda) {
    pcaScatter('chart-alr-pca', D.coda.alr.pc, 'PC1 (20.6%)', 'PC2 (20.1%)');
    pcaScatter('chart-clr-pca', D.coda.clr.pc, 'PC1 (6.9%)', 'PC2 (6.8%)');
  }

  /* ================= 10. explained variance ================= */
  var c10 = init('chart-evr');
  if (c10 && D.coda) {
    var n = Math.min(D.coda.alr.evr.length, D.coda.clr.evr.length, 12);
    var pcs = [];
    for (var pi = 1; pi <= n; pi++) pcs.push('PC' + pi);
    c10.setOption({
      animation: false,
      grid: { left: 58, right: 24, top: 34, bottom: 40 },
      tooltip: Object.assign({
        trigger: 'axis', axisPointer: { type: 'shadow' },
        formatter: function (ps) {
          var out = ps[0].name;
          ps.forEach(function (p) { out += '<br/>' + p.seriesName + ' ' + p.value.toFixed(1) + '%'; });
          return out;
        }
      }, tips),
      legend: { data: ['ALR', 'CLR'], top: 0, right: 6, textStyle: legendStyle },
      xAxis: ax({ type: 'category', data: pcs, axisLabel: { color: muted, fontFamily: font, fontSize: 10, interval: 0 } }),
      yAxis: ax({ type: 'value', name: '方差解释率 (%)', max: 23 }),
      series: [
        {
          name: 'ALR', type: 'bar', barWidth: '38%',
          data: D.coda.alr.evr.slice(0, n).map(function (v) { return v * 100; }),
          itemStyle: { color: accent2 },
          label: {
            show: true, color: muted, fontFamily: font, fontSize: 9, position: 'top',
            formatter: function (p) { return p.dataIndex < 3 ? p.value.toFixed(1) : ''; }
          }
        },
        {
          name: 'CLR', type: 'bar', barWidth: '38%',
          data: D.coda.clr.evr.slice(0, n).map(function (v) { return v * 100; }),
          itemStyle: { color: accent },
          label: {
            show: true, color: muted, fontFamily: font, fontSize: 9, position: 'top',
            formatter: function (p) { return p.dataIndex < 3 ? p.value.toFixed(1) : ''; }
          }
        }
      ]
    });
  }

  /* ================= 11. geometry consistency ================= */
  var c11 = init('chart-geom');
  if (c11 && D.coda && D.coda.geom) {
    var g = D.coda.geom;
    c11.setOption({
      animation: false,
      grid: { left: 62, right: 24, top: 34, bottom: 40 },
      tooltip: Object.assign({
        trigger: 'axis', axisPointer: { type: 'shadow' },
        formatter: function (ps) {
          var out = ps[0].name;
          ps.forEach(function (p) {
            out += '<br/>' + p.seriesName + ' ' + (
              p.seriesName === '距离相关系数' ? p.value.toFixed(3) : p.value.toFixed(3)
            );
          });
          return out;
        }
      }, tips),
      legend: { data: ['距离相关系数', '20 近邻 Jaccard'], top: 0, right: 6, textStyle: legendStyle },
      xAxis: ax({ type: 'category', data: g.map(function (r) { return r.building + ' 号楼'; }) }),
      yAxis: ax({ type: 'value', min: 0, max: 1, name: '一致性' }),
      series: [
        {
          name: '距离相关系数', type: 'bar', barWidth: '30%',
          data: g.map(function (r) { return r.dist_corr; }),
          itemStyle: { color: accent },
          label: { show: true, position: 'top', color: muted, fontFamily: font, fontSize: 10, formatter: function (p) { return p.value.toFixed(3); } }
        },
        {
          name: '20 近邻 Jaccard', type: 'bar', barWidth: '30%',
          data: g.map(function (r) { return r.knn_jaccard; }),
          itemStyle: { color: accent2 },
          label: { show: true, position: 'top', color: muted, fontFamily: font, fontSize: 10, formatter: function (p) { return p.value.toFixed(3); } }
        }
      ]
    });
  }

  /* ================= 12. PDP curves ================= */
  var c12 = init('chart-pdp');
  if (c12 && D.pdp) {
    var pal = [accent, accent2, '#3b8d80', '#c98a3c', muted];
    c12.setOption({
      animation: false,
      grid: { left: 60, right: 24, top: 44, bottom: 48 },
      tooltip: Object.assign({ trigger: 'axis' }, tips),
      legend: { type: 'scroll', top: 0, right: 4, textStyle: legendStyle },
      xAxis: ax({ type: 'value', name: '该指标权重', min: 0, max: 0.6 }),
      yAxis: ax({ type: 'value', name: '预测 ramping', min: 135, max: 180 }),
      series: D.pdp.curves.map(function (cur, idx) {
        return {
          name: cur.name, type: 'line',
          data: cur.values.map(function (v, j) { return [D.pdp.grid[j], v]; }),
          smooth: false, symbol: 'circle', symbolSize: 4,
          lineStyle: { width: 1.8, color: pal[idx % pal.length] },
          itemStyle: { color: pal[idx % pal.length] }
        };
      })
    });
  }

  /* ================= 13. PDP 2D ================= */
  var c13 = init('chart-pdp2d');
  if (c13 && D.pdp && D.pdp.grid2) {
    var g2 = D.pdp.grid2, cells2 = [], vals = [];
    for (var yi = 0; yi < g2.y.length; yi++) for (var xi = 0; xi < g2.x.length; xi++) {
      cells2.push([xi, yi, g2.z[yi][xi]]);
      vals.push(g2.z[yi][xi]);
    }
    c13.setOption({
      animation: false,
      grid: { left: 78, right: 28, top: 24, bottom: 86 },
      tooltip: Object.assign({
        formatter: function (p) {
          return 'b1_ramping = ' + g2.x[p.value[0]].toFixed(2) +
            '<br/>b1_degradation = ' + g2.y[p.value[1]].toFixed(2) +
            '<br/>预测 ramping = ' + p.value[2].toFixed(1);
        }
      }, tips),
      xAxis: ax({ type: 'category', data: g2.x.map(function (v) { return v.toFixed(2); }), name: 'b1_ramping', splitArea: { show: false } }),
      yAxis: ax({ type: 'category', data: g2.y.map(function (v) { return v.toFixed(2); }), name: 'b1_degradation', splitArea: { show: false } }),
      visualMap: {
        min: Math.min.apply(null, vals), max: Math.max.apply(null, vals), calculable: false,
        orient: 'horizontal', left: 'center', bottom: 8, itemWidth: 12, itemHeight: 100,
        textStyle: { color: muted, fontFamily: font, fontSize: 10 },
        inRange: { color: [accent, '#eef3f1', accent2] }
      },
      series: [{
        type: 'heatmap', data: cells2, splitArea: { show: false },
        label: { show: true, color: ink, fontFamily: font, fontSize: 8, formatter: function (p) { return p.value[2].toFixed(0); } }
      }]
    });
  }

  /* ================= 14. probe comparison ================= */
  var c14 = init('chart-probe');
  if (c14 && D.coda && D.coda.probe) {
    var pr = D.coda.probe;
    c14.setOption({
      animation: false,
      grid: { left: 56, right: 24, top: 38, bottom: 40 },
      tooltip: Object.assign({ trigger: 'axis', axisPointer: { type: 'shadow' } }, tips),
      legend: { data: ['Ridge', 'GBDT'], top: 0, right: 6, textStyle: legendStyle },
      xAxis: ax({ type: 'category', data: ['原始权重', 'ALR', 'CLR'] }),
      yAxis: ax({ type: 'value', name: 'CV-MAE', min: 0, max: 12 }),
      series: [
        {
          name: 'Ridge', type: 'bar', barWidth: '30%',
          data: [pr.raw, pr.alr, pr.clr],
          itemStyle: { color: accent + '99' },
          label: { show: true, position: 'top', color: muted, fontFamily: font, fontSize: 10, formatter: function (p) { return p.value.toFixed(2); } }
        },
        {
          name: 'GBDT', type: 'bar', barWidth: '30%',
          data: [pr.raw_gbdt, pr.alr_gbdt, pr.clr_gbdt],
          itemStyle: { color: accent2 + 'cc' },
          label: { show: true, position: 'top', color: muted, fontFamily: font, fontSize: 10, formatter: function (p) { return p.value.toFixed(2); } }
        }
      ]
    });
  }
})();
