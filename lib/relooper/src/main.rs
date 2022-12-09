mod cfg;
mod debug_view;
mod re_graph;
mod relooper;
mod traversal;

use crate::cfg::{Cfg, CfgLabel};
use crate::re_graph::{ReBlock, ReGraph, ReLabel};
use crate::traversal::graph::dfs::Dfs;

pub fn main() {
    use std::fs::File;

    let graph = Cfg::from(vec![
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (1, 5),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 8),
        (4, 9),
        (8, 9),
        (8, 5),
    ]);
    // let graph = Cfg::from(vec![(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (6, 7)]);

    let mut f_cfg = File::create("cfg.dot").unwrap();
    dot::render(&graph, &mut f_cfg).unwrap();

    let dfs: Vec<_> = Dfs::start_from(0 as CfgLabel, |&n| graph.children(n).into_iter()).collect();
    println!("dfs: {:?}", dfs);

    // let re_builder = ReBuilder::create(&graph, 0);
    // let re_graph = re_builder.reloop();

    // let mut f_relooped = File::create("relooped.dot").unwrap();
    // dot::render(&re_graph, &mut f_relooped).unwrap();

    // let start = 0 as CfgLabel;
    // let res_b = BfsGraph::start_from(&start).traverse(|x| graph.0.get(x).into_iter().flatten());
    // println!("Bfs:{:?}", res_b);
    // let res_d = DfsGraph::start_from(&start).traverse(|x| graph.0.get(x).into_iter().flatten());
    // println!("Dfs:{:?}", res_d);
}
