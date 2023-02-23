use std::collections::{BTreeSet, HashSet, VecDeque};
use std::hash::Hash;

pub struct Dfs<T, ChFun> {
    visited: HashSet<T>,
    queue: VecDeque<T>,
    get_children: ChFun,
}

impl<T, ChIt, ChFun> Dfs<T, ChFun>
where
    ChIt: IntoIterator<Item = T>,
    ChFun: FnMut(&T) -> ChIt,
{
    pub fn start_iter<I: IntoIterator<Item = T>>(iter: I, get_children: ChFun) -> Self {
        Dfs {
            visited: HashSet::new(),
            queue: VecDeque::from_iter(iter),
            get_children,
        }
    }

    pub fn start_from(item: T, get_children: ChFun) -> Self {
        Self::start_iter(Some(item).into_iter(), get_children)
    }

    pub fn start_from_except(item: &T, mut get_children: ChFun) -> Self {
        Self::start_iter(get_children(item), get_children)
    }
}

impl<T, ChIt, ChFun> Iterator for Dfs<T, ChFun>
where
    T: Hash + Eq + Copy,
    ChIt: IntoIterator<Item = T>,
    ChFun: FnMut(&T) -> ChIt,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop_back().map(|current| {
            let children = (self.get_children)(&current)
                .into_iter()
                .filter(|c| !self.visited.contains(c))
                .collect::<HashSet<_>>();
            for &c in &children {
                self.visited.insert(c);
            }
            self.queue.extend(children.into_iter());

            current
        })
    }
}


pub fn dfs_post_hashable<T, ChIt, ChFun>(start: T, get_children: &mut ChFun) -> Vec<T>
where
    T: Hash + Eq + Copy,
    ChIt: IntoIterator<Item = T>,
    ChFun: FnMut(&T) -> ChIt,
{
    let mut visited: HashSet<T> = HashSet::from([start]);
    let mut res: Vec<T> = Vec::new();
    let mut stack = vec![start];

    while let Some(current) = stack.pop() {
        for chld in get_children(&current) {
            if !visited.contains(&chld) {
                visited.insert(chld);
                stack.push(chld);
            } 
        }
        res.push(current);
    }

    res.reverse();
    res
}


pub fn dfs_post_comparable<T, ChIt, ChFun>(start: T, get_children: &mut ChFun) -> Vec<T>
where
    T: Ord + Eq + Copy,
    ChIt: IntoIterator<Item = T>,
    ChFun: FnMut(&T) -> ChIt,
{
    let mut visited: BTreeSet<T> = BTreeSet::from([start]);
    let mut res: Vec<T> = Vec::new();
    let mut stack = vec![start];

    while let Some(current) = stack.pop() {
        for chld in get_children(&current) {
            if !visited.contains(&chld) {
                visited.insert(chld);
                stack.push(chld);
            } 
        }
        res.push(current);
    }

    res.reverse();
    res
}

use crate::graph::cfg::{Cfg, CfgEdge};
use std::collections::HashMap;


    #[test]
    pub fn test_dfs() {
        let cfg = Cfg::from_edges(
            0,
            &HashMap::from([
                (0, CfgEdge::Cond(1, 2)),
                (1, CfgEdge::Uncond(2)),
                (2, CfgEdge::Uncond(3)),
                (3, CfgEdge::Cond(4, 5)),
                (4, CfgEdge::Uncond(6)),
                (5, CfgEdge::Uncond(6)),
            ]),
        );
        let comparable = dfs_post_comparable(&0, &mut |node| cfg.children(node));
        let hashable = dfs_post_hashable(&0, &mut |node| cfg.children(node));

        let mut to_write: Vec<String> = vec!["comparable:\n".to_string()];
        to_write.push(comparable.into_iter().map(|x|x.to_string()).collect::<Vec<_>>().join(" "));
        to_write.push("\nhashable\n".to_string());
        to_write.push(hashable.into_iter().map(|x|x.to_string()).collect::<Vec<_>>().join(" "));

        std::fs::write("dfs.txt", to_write.join("\n")).unwrap();
    }
