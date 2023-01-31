use crate::graph::cfg::CfgEdge::{Cond, Terminal, Uncond};
use crate::traversal::graph::bfs::Bfs;
use anyhow::ensure;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::once;

mod cfg_parsing;

pub trait CfgLabel: Copy + Hash + Eq + Ord + Sized {}

impl<T: Copy + Hash + Eq + Ord + Sized> CfgLabel for T {}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum CfgEdge<TLabel> {
    Uncond(TLabel),
    Cond(TLabel, TLabel),
    Terminal,
}

impl<TLabel> CfgEdge<TLabel> {
    pub fn to_vec(&self) -> Vec<&TLabel> {
        match self {
            Uncond(u) => vec![u],
            Cond(cond, fallthrough) => vec![cond, fallthrough],
            Terminal => vec![],
        }
    }

    fn as_ref(&self) -> CfgEdge<&TLabel> {
        match *self {
            Uncond(ref to) => Uncond(to),
            Cond(ref t, ref f) => Cond(t, f),
            Terminal => Terminal,
        }
    }

    pub(crate) fn map<'a, U, F: Fn(&'a TLabel) -> U>(&'a self, mapping: F) -> CfgEdge<U> {
        match self {
            Uncond(t) => Uncond(mapping(t)),
            Cond(t, f) => Cond(mapping(t), mapping(f)),
            Terminal => Terminal,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cfg<TLabel> {
    pub(crate) entry: TLabel,
    pub(crate) out_edges: HashMap<TLabel, CfgEdge<TLabel>>,
}

impl<T: Eq + Hash> Cfg<T> {
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
}

impl<TLabel: Eq + Hash + Copy> Cfg<TLabel> {
    pub fn from_edges(
        entry: TLabel,
        edges: &HashMap<TLabel, CfgEdge<TLabel>>,
    ) -> Result<Self, anyhow::Error> {
        let mut out_edges = HashMap::new();
        let mut nodes = HashSet::new();
        for (&from, &edge) in edges.iter() {
            let old_val = out_edges.insert(from, edge);

            ensure!(old_val.is_none(), "repeating source node");

            nodes.insert(from);
            nodes.extend(edge.to_vec());
        }

        for n in nodes {
            out_edges.entry(n).or_insert(Terminal);
        }

        Ok(Self { entry, out_edges })
    }

    pub fn from_vec(
        entry: TLabel,
        edges: &[(TLabel, CfgEdge<TLabel>)],
    ) -> Result<Self, anyhow::Error> {
        let edges_map: HashMap<TLabel, CfgEdge<TLabel>> = edges.iter().copied().collect();
        Self::from_edges(entry, &edges_map)
    }
}

impl<TLabel: CfgLabel> Cfg<TLabel> {
    pub fn nodes(&self) -> HashSet<&TLabel> {
        self.out_edges
            .iter()
            .flat_map(|(from, to)| once(from).chain(to.to_vec()))
            .collect()
    }

    pub fn edge(&self, label: &TLabel) -> &CfgEdge<TLabel> {
        self.out_edges
            .get(label)
            .expect("any node should have outgoing edges")
    }

    pub fn children(&self, label: &TLabel) -> HashSet<&TLabel> {
        self.out_edges
            .get(label)
            .into_iter()
            .flat_map(|edge| edge.to_vec())
            .collect()
    }

    pub fn in_edges(&self) -> HashMap<TLabel, HashSet<TLabel>> {
        let mut in_edges: HashMap<TLabel, HashSet<TLabel>> = HashMap::default();

        for (&from, to_edge) in &self.out_edges {
            for &to in to_edge.to_vec() {
                in_edges.entry(to).or_default().insert(from);
            }
        }

        in_edges
    }

    pub fn add_edge(&mut self, from: TLabel, edge: CfgEdge<TLabel>) {
        assert!(self.out_edges.insert(from, edge).is_none());
    }

    pub fn remove_edge(&mut self, from: TLabel, edge: CfgEdge<TLabel>) {
        let removed_edge = self.out_edges.remove(&from);
        assert!(removed_edge == Some(edge));
    }

    pub fn add_edge_or_promote(&mut self, from: TLabel, to: TLabel) {
        match self.out_edges.remove(&from) {
            None | Some(Terminal) => self.out_edges.insert(from, Uncond(to)),
            Some(Uncond(uncond)) => self.out_edges.insert(from, Cond(to, uncond)),
            _ => panic!("edge (should be absent) or (shouldn't be `Cond`)"),
        };
    }

    fn reachable_nodes(&self) -> HashSet<TLabel> {
        Bfs::start_from(self.entry, |label| {
            self.children(label).into_iter().copied()
        })
        .collect()
    }

    pub fn strip_unreachable(&mut self) {
        let nodes: HashSet<TLabel> = self.nodes().into_iter().copied().collect();
        let reachable: HashSet<TLabel> = self.reachable_nodes();
        for unreachable in nodes.difference(&reachable) {
            self.out_edges.remove(unreachable);
        }
    }
}
