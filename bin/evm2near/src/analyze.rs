// This is free and unencumbered software released into the public domain.

use evm_rs::{Opcode, Program};
use relooper::graph::cfg::{Cfg, CfgEdge};
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    ops::Range,
};


/// This struct represents offset of instruction in EVM bytecode.
/// Mostly used for easy mapping between generated wasm instruction 
/// and its EVM image. Also it's a good label for CFG block
/// because it's unique for each block and provide some extra information
/// about the block
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Offs(pub usize);

/// This struct represents the serial number of instruction in EVM bytecode.
/// Same as Offs struct it is very useful to know this number of instruction 
/// during the debug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Idx(pub usize);

impl Display for Offs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:x}", self.0)
    }
}

impl Display for Idx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// struct BlockStart(Offs, Idx, bool);
struct BlockStart {
    offs: Offs,
    idx: Idx,
    is_jumpdest: bool,
}

impl Debug for BlockStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "offs: {}, idx: {}, jumpdest? {}", self.offs, self.idx, self.is_jumpdest)
    }
}

#[derive(Debug)]
pub struct BasicCfg {
    pub cfg: Cfg<Offs>,
    pub node_info: HashMap<Offs, (bool, bool)>,
    pub code_ranges: HashMap<Offs, Range<Idx>>,
}

pub fn basic_cfg(program: &Program) -> BasicCfg {
    let mut cfg = Cfg::new(Offs(0));
    let mut node_info: HashMap<Offs, (bool, bool)> = Default::default(); // label => (is_jumpdest, is_dynamic);
    let mut code_ranges: HashMap<Offs, Range<Idx>> = Default::default();

    let mut curr_offs = Offs(0);
    let mut block_start: Option<BlockStart> = None;
    let mut prev_op: Option<&Opcode> = None;
    let mut next_idx = Idx(1);

    for (curr_idx, op) in program.0.iter().enumerate().map(|(i, op)| (Idx(i), op)) {
        next_idx = Idx(curr_idx.0 + 1);
        let next_offs = Offs(curr_offs.0 + op.size());

        use Opcode::*;
        block_start = match op {
            JUMP | JUMPI => {
                let BlockStart{offs: start_offs,idx: start_idx,is_jumpdest: is_jmpdest} =
                    block_start.expect("block should be present at any jump opcode");

                let label = match prev_op {
                    Some(PUSH1(addr)) => Some(Offs(usize::from(*addr))),
                    Some(PUSHn(_, addr, _)) => Some(Offs(addr.as_usize())),
                    Some(_) => None,
                    None => unreachable!(),
                };
                let is_dyn = match label {
                    Some(l) => {
                        let edge = if op == &JUMP {
                            CfgEdge::Uncond(l)
                        } else {
                            CfgEdge::Cond(l, next_offs)
                        };
                        cfg.add_edge(start_offs, edge);
                        false
                    }
                    None => true,
                };
                node_info.insert(start_offs, (is_jmpdest, is_dyn));
                code_ranges.insert(start_offs, start_idx..next_idx);
                if is_jmpdest && is_dyn {
                    cfg.add_node(start_offs);
                }

                None
            }
            JUMPDEST => {
                if let Some(BlockStart{offs: start_offs, idx: start_idx, is_jumpdest: is_jmpdest}) = block_start {
                    let edge = CfgEdge::Uncond(curr_offs);
                    cfg.add_edge(start_offs, edge);
                    node_info.insert(start_offs, (is_jmpdest, false));
                    code_ranges.insert(start_offs, start_idx..curr_idx);
                }

                Some(BlockStart{offs: curr_offs, idx: curr_idx, is_jumpdest: true})
            }
            _ => {
                let bs @ BlockStart{offs: start_offs, idx: start_idx, is_jumpdest: is_jmpdest} =
                    block_start.unwrap_or(BlockStart{offs: curr_offs, idx: curr_idx, is_jumpdest: false});

                if op.is_halt() {
                    cfg.add_edge(bs.offs, CfgEdge::Terminal);
                    node_info.insert(start_offs, (is_jmpdest, false));
                    code_ranges.insert(start_offs, start_idx..next_idx);
                    None
                } else {
                    Some(bs)
                }
            }
        };

        curr_offs = next_offs;
        prev_op = Some(op);
    }

    if let Some(BlockStart{offs: start_offs, idx: start_idx, is_jumpdest: is_jmpdest}) = block_start {
        node_info.insert(start_offs, (is_jmpdest, false));
        code_ranges.insert(start_offs, start_idx..next_idx);
    }

    BasicCfg {
        cfg,
        node_info,
        code_ranges,
    }
}
