use crate::graph::cfg::CfgEdge::{Cond, Terminal, Uncond};
use crate::graph::cfg::{Cfg, CfgDescr, CfgEdge, CfgLabel};
use anyhow::{ensure, format_err};
use std::collections::HashMap;
use std::hash::Hash;
use std::str::FromStr;

impl<TLabel: FromStr> FromStr for CfgEdge<TLabel> {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (first_str, maybe_second) = {
            let mut split_i = value.split(' ');
            (split_i.next().unwrap(), split_i.next())
        };

        match maybe_second {
            None => {
                let a = TLabel::from_str(first_str)
                    .map(Uncond)
                    .map_err(|_e| anyhow::Error::msg("label parsing error")); //TODO unable to find solution for non-std-err conversion to anyhow error
                a
            }
            Some(uncond_str) => {
                let cond = TLabel::from_str(first_str)
                    .map_err(|_e| anyhow::Error::msg("label parsing error"))?;
                let uncond = TLabel::from_str(uncond_str)
                    .map_err(|_e| anyhow::Error::msg("label parsing error"))?;
                Ok(Cond(cond, uncond))
            }
        }
    }
}

impl<TLabel: FromStr + Eq + Hash> TryFrom<&Vec<String>> for CfgDescr<TLabel> {
    type Error = anyhow::Error;

    fn try_from(strings: &Vec<String>) -> Result<Self, Self::Error> {
        ensure!(
            strings.len() >= 2,
            "well-formed cfg should contain entry line and at least one edge"
        );

        let entry_str = strings.first().unwrap();
        let entry =
            TLabel::from_str(entry_str).map_err(|e| anyhow::Error::msg("label parsing error"))?;

        let mut edges = HashMap::with_capacity(strings.len() - 1);

        for edge_str in &strings[1..] {
            let (from, edge) = edge_str
                .split_once(' ')
                .ok_or_else(|| format_err!("invalid label-edge format".to_string()))?;
            let from =
                TLabel::from_str(from).map_err(|e| anyhow::Error::msg("label parsing error"))?;
            let edge = CfgEdge::from_str(edge)?;
            edges.insert(from, edge);
        }

        Ok(Self { entry, edges })
    }
}
