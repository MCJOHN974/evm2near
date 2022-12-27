use crate::graph::cfg::{CfgEdge, CfgLabel};
use crate::graph::relooper::{ReBlock, ReSeq};
use crate::graph::EnrichedCfg;
use std::fmt::format;
use std::thread::current;

impl EnrichedCfg {
    fn labels(&self, n: CfgLabel) -> String {
        let mut res = "".to_string();
        if self.loop_nodes.contains(&n) {
            res += "l";
        }
        if self.if_nodes.contains(&n) {
            res += "i";
        }
        if self.merge_nodes.contains(&n) {
            res += "m";
        }

        res
    }

    pub fn cfg_to_dot(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push("subgraph cluster_cfg { label=\"cfg\";".to_string());
        lines.push("nstart[label=\"start\"]".to_string());
        lines.push("nend[label=\"end\"]".to_string());

        let mut edges: Vec<String> = Vec::new();
        for n in self.cfg.nodes() {
            lines.push(format!("n{n}[label=\"{n} {}\"];", self.labels(n)));
            match self.cfg.edge(n) {
                CfgEdge::Uncond(u) => {
                    edges.push(format!("n{n} -> n{u};"));
                }
                CfgEdge::Cond(t, f) => {
                    edges.push(format!("n{n} -> n{t}[style=\"dashed\"];"));
                    edges.push(format!("n{n} -> n{f};"));
                }
                CfgEdge::Terminal => {
                    edges.push(format!("n{n} -> nend;"));
                }
            }
        }
        lines.push(format!("nstart -> n{}", self.entry));
        lines.extend(edges);
        lines.push("}".to_string());

        lines.join("\n")
    }

    pub fn dom_to_dot(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        lines.push("subgraph cluster_dom { label=\"dom\"; edge [dir=\"back\"];".to_string());
        for n in self.cfg.nodes() {
            lines.push(format!("d{n}[label=\"{n}\"];"));
        }
        for (&n, &d) in &self.domination.dominated {
            lines.push(format!("d{d} -> d{n};"));
        }
        lines.push("}".to_string());

        lines.join("\n")
    }
}

impl ReSeq {
    fn to_dot_inner(&self, current_id: usize, back_branches: &Vec<usize>) -> (usize, Vec<String>) {
        let mut res: Vec<String> = Vec::new();
        let id = self
            .0
            .iter()
            .fold(current_id, |current_id, block| match block {
                ReBlock::Block(next) | ReBlock::Loop(next) => {
                    let mut back = back_branches.clone();
                    back.push(current_id);

                    let bstr = if let ReBlock::Block(_) = block {
                        "Block".to_string()
                    } else {
                        "Loop".to_string()
                    };
                    res.push(format!("r{current_id}[label=\"{bstr} {current_id}\"];"));

                    let ch_id = current_id + 1;
                    let (ch_last_id, ch_str) = next.to_dot_inner(ch_id, &back);
                    res.extend(ch_str);

                    res.push(format!("r{current_id} -> r{ch_id};"));

                    let next_id = ch_last_id + 1;
                    res.push(format!("r{current_id} -> r{next_id}[style=\"bold\"];"));

                    next_id
                }
                ReBlock::If(t, f) => {
                    let mut back = back_branches.clone();
                    back.push(current_id);

                    res.push(format!("r{current_id}[label=\"If {current_id}\"];"));

                    let t_ch_id = current_id + 1;
                    let (t_id, t_str) = t.to_dot_inner(t_ch_id, &back);
                    res.push(format!("r{current_id} -> r{t_ch_id}[style=\"dashed\"];"));

                    let f_ch_id = t_id + 1;
                    let (f_id, f_str) = f.to_dot_inner(f_ch_id, &back);
                    res.push(format!("r{current_id} -> r{f_ch_id};"));

                    res.extend(t_str);
                    res.extend(f_str);

                    let next_id = f_id + 1;
                    res.push(format!("r{current_id} -> r{next_id}[style=\"bold\"];"));

                    next_id
                }
                ReBlock::Actions(label) => {
                    res.push(format!(
                        "r{current_id}[label=\"{label} Actions {current_id}\"];"
                    ));

                    let next_id = current_id + 1;
                    res.push(format!("r{current_id} -> r{next_id}[style=\"bold\"];"));

                    next_id
                }
                ReBlock::Br(jmp) => {
                    res.push(format!("r{current_id}[label=\"Br {current_id}\"];"));

                    let branch_to = back_branches
                        .get(back_branches.len() - 1 - jmp)
                        .expect("unexpected branch");
                    res.push(format!("r{current_id} -> r{branch_to}[constraint=false]"));

                    current_id + 1
                }
                ReBlock::Return => {
                    res.push(format!("r{current_id}[label=\"Return {current_id}\"];"));

                    current_id + 1
                }
            });
        (id, res)
    }

    pub fn to_dot(&self) -> String {
        let (_id, strs) = self.to_dot_inner(0, &vec![]);
        let mut lines: Vec<String> =
            vec!["subgraph cluster_relooped { label=\"relooped\";".to_string()];
        lines.extend(strs);
        lines.push("}".to_string());

        lines.join("\n")
    }
}
