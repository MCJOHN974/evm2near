use crate::graph::cfg::CfgEdge::{Cond, Switch, Terminal, Uncond};
use crate::traversal::graph::bfs::Bfs;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

mod cfg_parsing;

pub trait CfgLabel: Copy + Hash + Eq + Ord + Debug {}

impl<T: Copy + Hash + Eq + Ord + Debug> CfgLabel for T {}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum CfgEdge<TLabel> {
    Uncond(TLabel),
    Cond(TLabel, TLabel),
    Switch(Vec<(usize, TLabel)>),
    Terminal,
}

impl<TLabel> CfgEdge<TLabel> {
    pub fn iter(&self) -> CfgEdgeIter<&TLabel> {
        match self {
            Uncond(u) => CfgEdgeIter::Fixed(CfgEdgeFixedIter {
                inner: [Some(u), None],
                index: 0,
            }),
            Cond(cond, fallthrough) => CfgEdgeIter::Fixed(CfgEdgeFixedIter {
                inner: [Some(cond), Some(fallthrough)],
                index: 0,
            }),
            Switch(v) => {
                if v.len() <= 2 {
                    let inner = [v.get(0).map(|(_, l)| l), v.get(1).map(|(_, l)| l)];
                    CfgEdgeIter::Fixed(CfgEdgeFixedIter { inner, index: 0 })
                } else {
                    CfgEdgeIter::Flexible(v.iter().map(|(_, l)| l).collect())
                }
            }
            Terminal => CfgEdgeIter::Fixed(CfgEdgeFixedIter {
                inner: [None, None],
                index: 0,
            }),
        }
    }

    pub(crate) fn apply<F: Fn(&TLabel) -> TLabel>(&mut self, mapping: F) {
        match self {
            Self::Uncond(t) => {
                *self = Self::Uncond(mapping(t));
            }
            Self::Cond(t, f) => {
                *self = Self::Cond(mapping(t), mapping(f));
            }
            Self::Switch(v) => {
                for (_, x) in v {
                    *x = mapping(x)
                }
            }
            Self::Terminal => {}
        }
    }

    pub(crate) fn map<'a, U, F: Fn(&'a TLabel) -> U>(&'a self, mapping: F) -> CfgEdge<U> {
        match self {
            Self::Uncond(t) => Uncond(mapping(t)),
            Self::Cond(t, f) => Cond(mapping(t), mapping(f)),
            Self::Switch(v) => Switch(v.iter().map(|(u, x)| (*u, mapping(x))).collect()),
            Self::Terminal => Terminal,
        }
    }
}

/// A enum which enables iterating over the nodes that make up a `CfgEdge`.
/// Internally it stores the data as a 2-array for fixed-size edges as opposed to a `Vec` to avoid heap allocation.
/// In case of `Switch` variant, `VecDeque` is used if there is more than 2 children.
#[derive(Debug, Clone, Copy)]
pub struct CfgEdgeFixedIter<T> {
    inner: [Option<T>; 2],
    index: usize,
}

#[derive(Debug, Clone)]
pub enum CfgEdgeIter<T> {
    Fixed(CfgEdgeFixedIter<T>),
    Flexible(VecDeque<T>),
}

impl<T> Iterator for CfgEdgeIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Fixed(fixed) => {
                if fixed.index >= 2 {
                    return None;
                }
                let result = fixed.inner[fixed.index].take();
                fixed.index += 1;
                result
            }
            Self::Flexible(deque) => deque.pop_front(),
        }
    }
}

impl<T: Copy> IntoIterator for CfgEdge<T> {
    type Item = T;

    type IntoIter = CfgEdgeIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Uncond(u) => CfgEdgeIter::Fixed(CfgEdgeFixedIter {
                inner: [Some(u), None],
                index: 0,
            }),
            Cond(cond, fallthrough) => CfgEdgeIter::Fixed(CfgEdgeFixedIter {
                inner: [Some(cond), Some(fallthrough)],
                index: 0,
            }),
            Switch(v) => {
                if v.len() <= 2 {
                    let inner = [v.get(0).map(|&(_, l)| l), v.get(1).map(|&(_, l)| l)];
                    CfgEdgeIter::Fixed(CfgEdgeFixedIter { inner, index: 0 })
                } else {
                    CfgEdgeIter::Flexible(v.into_iter().map(|(_, l)| l).collect())
                }
            }
            Terminal => CfgEdgeIter::Fixed(CfgEdgeFixedIter {
                inner: [None, None],
                index: 0,
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cfg<TLabel> {
    pub(crate) entry: TLabel,
    out_edges: HashMap<TLabel, CfgEdge<TLabel>>,
}

impl<T> Cfg<T> {
    pub fn edges(&self) -> &HashMap<T, CfgEdge<T>> {
        &self.out_edges
    }
}

impl<T: Eq + Hash + Clone> Cfg<T> {
    pub fn new(entry: T) -> Cfg<T> {
        Self {
            entry,
            out_edges: Default::default(),
        }
    }

    pub fn map_label<'a, M, U: Eq + Hash>(&'a self, mapping: M) -> Cfg<U>
    where
        M: Fn(&'a T) -> U,
    {
        let out_edges: HashMap<U, CfgEdge<U>> = self
            .out_edges
            .iter()
            .map(|(from, e)| (mapping(from), e.map(&mapping)))
            .collect();

        Cfg {
            entry: mapping(&self.entry),
            out_edges,
        }
    }

    pub fn to_borrowed(&self) -> Cfg<&T> {
        self.map_label(|l| l)
    }

    pub fn nodes(&self) -> HashSet<&T> {
        self.out_edges.keys().collect()
    }

    pub fn children(&self, label: &T) -> HashSet<&T> {
        self.out_edges
            .get(label)
            .into_iter()
            .flat_map(|edge| edge.iter())
            .collect()
    }

    fn check_previous_edge(edge: Option<CfgEdge<T>>) {
        match edge {
            None | Some(Terminal) => {}
            _ => panic!("adding edge over already present one"),
        }
    }

    pub fn add_edge(&mut self, from: T, edge: CfgEdge<T>) {
        let out_edges = &mut self.out_edges;
        for n in edge.iter() {
            if !out_edges.contains_key(n) {
                // The clone here is required because we use `edge` again in the insert below
                out_edges.insert(n.clone(), Terminal);
            }
        }

        let prev_edge = out_edges.insert(from, edge);
        Self::check_previous_edge(prev_edge);
    }

    pub fn add_node(&mut self, n: T) {
        let prev_edge = self.out_edges.insert(n, Terminal);
        Self::check_previous_edge(prev_edge);
    }

    pub fn remove_edge(&mut self, from: T, edge: &CfgEdge<T>) {
        let removed_edge = self.out_edges.remove(&from);
        assert!(removed_edge.as_ref() == Some(edge));
    }

    pub fn add_edge_or_promote(&mut self, from: T, to: T) {
        match self.out_edges.remove(&from) {
            None | Some(Terminal) => self.out_edges.insert(from, Uncond(to)),
            Some(Uncond(uncond)) => self.out_edges.insert(from, Cond(to, uncond)),
            _ => panic!("edge (should be absent) or (shouldn't be `Cond`)"),
        };
    }

    pub fn edge(&self, label: &T) -> &CfgEdge<T> {
        self.out_edges
            .get(label)
            .expect("any node should have outgoing edges")
    }

    pub fn edge_mut(&mut self, label: &T) -> &mut CfgEdge<T> {
        self.out_edges
            .get_mut(label)
            .expect("any node should have outgoing edges")
    }
}

impl<TLabel: Eq + Hash + Copy> Cfg<TLabel> {
    pub fn from_edges(entry: TLabel, edges: HashMap<TLabel, CfgEdge<TLabel>>) -> Self {
        let mut cfg = Cfg::new(entry);
        for (from, edge) in edges.into_iter() {
            cfg.add_edge(from, edge);
        }

        cfg
    }
}

impl<TLabel: CfgLabel> Cfg<TLabel> {
    pub fn in_edges(&self) -> HashMap<TLabel, HashSet<TLabel>> {
        let mut in_edges: HashMap<TLabel, HashSet<TLabel>> = HashMap::default();

        for (&from, to_edge) in &self.out_edges {
            for &to in to_edge.iter() {
                in_edges.entry(to).or_default().insert(from);
            }
        }

        in_edges
    }

    fn reachable_nodes(&self) -> HashSet<&TLabel> {
        Bfs::start_from(&self.entry, |label| self.children(label)).collect()
    }

    pub fn strip_unreachable(&mut self) {
        let unreachable_nodes: HashSet<TLabel> = self
            .nodes()
            .difference(&self.reachable_nodes())
            .into_iter()
            .map(|n| **n)
            .collect();
        for unreachable in unreachable_nodes {
            self.out_edges.remove(&unreachable);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_edge_iter() {
        test_cfg_edge_iter_inner(Vec::new());
        test_cfg_edge_iter_inner(vec![7]);
        test_cfg_edge_iter_inner(vec![123, 456]);
    }

    fn test_cfg_edge_iter_inner(input: Vec<usize>) {
        let edge = cfg_edge_from_slice(&input);

        let reconstructed: Vec<usize> = edge.iter().copied().collect();
        assert_eq!(input, reconstructed);

        let reconstructed: Vec<usize> = edge.into_iter().collect();
        assert_eq!(input, reconstructed);
    }

    fn cfg_edge_from_slice<T: Copy>(values: &[T]) -> CfgEdge<T> {
        match values {
            [] => CfgEdge::Terminal,
            [x] => CfgEdge::Uncond(*x),
            [x, y] => CfgEdge::Cond(*x, *y),
            _ => panic!("cfg_edge_from_slice: Slice must have two or fewer values!"),
        }
    }
}
