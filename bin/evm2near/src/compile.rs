// This is free and unencumbered software released into the public domain.

use std::{collections::HashMap, convert::TryInto, io::Write};

use evm_rs::{parse_opcode, Opcode, Program};
use parity_wasm::{
    builder::{FunctionBuilder, ModuleBuilder, SignatureBuilder},
    elements::{
        BlockType, ElementSegment, ExportEntry, FuncBody, ImportCountType, InitExpr, Instruction,
        Instructions, Internal, Local, Module, TableType, Type, ValueType,
    },
};
use relooper::graph::relooper::ReBlock;
use relooper::graph::{caterpillar::CaterpillarLabel, relooper::ReSeq, supergraph::SLabel};

use crate::{
    abi::Functions,
    analyze::{basic_cfg, relooped_cfg, BasicCfg, EvmLabel},
    config::CompilerConfig,
    encode::encode_push,
};

const TABLE_OFFSET: i32 = 0x1000;

fn evm_idx_to_offs(program: &Program) -> HashMap<usize, usize> {
    // let mut opcode_lines: Vec<String> = vec![];
    let mut idx2offs: HashMap<usize, usize> = Default::default();
    program
        .0
        .iter()
        .enumerate()
        .fold(0_usize, |offs, (cnt, opcode)| {
            // opcode_lines.push(format!("0x{:02x}\t{}", offs, opcode));
            idx2offs.insert(cnt, offs);
            offs + opcode.size()
        });
    // std::fs::write("opcodes.evm", opcode_lines.join("\n")).expect("fs error");
    idx2offs
}

pub fn compile(
    input_program: &Program,
    input_abi: Option<Functions>,
    runtime_library: Module,
    config: CompilerConfig,
) -> Module {
    let mut compiler = Compiler::new(runtime_library, config);
    compiler.emit_wasm_start();
    compiler.emit_evm_start();
    compiler.compile_cfg(input_program);
    compiler.emit_abi_execute();
    let abi_data = compiler.emit_abi_methods(input_abi).unwrap();

    let mut output_module = compiler.builder.build();

    let tables = output_module.table_section_mut().unwrap().entries_mut();
    tables[0] = TableType::new(0xFFFF, Some(0xFFFF)); // grow the table to 65,535 elements

    // Overwrite the `_abi_buffer` data segment in evmlib with the ABI data
    // (function parameter names and types) for all public Solidity contract
    // methods:
    let abi_buffer_ptr: usize = compiler.abi_buffer_off.try_into().unwrap();
    for data in output_module.data_section_mut().unwrap().entries_mut() {
        let min_ptr: usize = match data.offset().as_ref().unwrap().code() {
            [Instruction::I32Const(off), Instruction::End] => (*off).try_into().unwrap(),
            _ => continue, // skip any nonstandard data segments
        };
        let max_ptr: usize = min_ptr + data.value().len();
        if abi_buffer_ptr >= min_ptr && abi_buffer_ptr < max_ptr {
            let min_off = abi_buffer_ptr - min_ptr;
            let max_off = min_off + abi_data.len();
            assert!(min_ptr + max_off <= max_ptr);
            data.value_mut()[min_off..max_off].copy_from_slice(&abi_data);
            break; // found it
        }
    }

    output_module
}

type DataOffset = i32;
type FunctionIndex = u32;
type TypeIndex = u32;

struct Compiler {
    config: CompilerConfig,
    abi_buffer_off: DataOffset,
    abi_buffer_len: usize,
    op_table: HashMap<Opcode, FunctionIndex>,
    function_type: TypeIndex,
    evm_start_function: FunctionIndex,     // _evm_start
    evm_init_function: FunctionIndex,      // _evm_init
    evm_call_function: FunctionIndex,      // _evm_call
    evm_exec_function: FunctionIndex,      // _evm_exec
    evm_post_exec_function: FunctionIndex, // _evm_post_exec
    evm_pop_function: FunctionIndex,       // _evm_pop_u32
    evm_push_function: FunctionIndex,      // _evm_push_u32
    evm_burn_gas: FunctionIndex,           // _evm_burn_gas
    evm_pc_function: FunctionIndex,        // _evm_set_pc
    function_import_count: usize,
    builder: ModuleBuilder,
}

impl Compiler {
    /// Instantiates a new compiler state.
    fn new(runtime_library: Module, config: CompilerConfig) -> Compiler {
        Compiler {
            config,
            abi_buffer_off: find_abi_buffer(&runtime_library).unwrap(),
            abi_buffer_len: 0xFFFF, // TODO: ensure this matches _abi_buffer.len() in evmlib
            op_table: make_op_table(&runtime_library),
            // jump_table: HashMap::new(),
            function_type: find_runtime_function_type(&runtime_library).unwrap(),
            evm_start_function: 0, // filled in during emit_start()
            evm_init_function: find_runtime_function(&runtime_library, "_evm_init").unwrap(),
            evm_call_function: find_runtime_function(&runtime_library, "_evm_call").unwrap(),
            evm_post_exec_function: find_runtime_function(&runtime_library, "_evm_post_exec")
                .unwrap(),
            evm_exec_function: 0, // filled in during compile_cfg()
            evm_pop_function: find_runtime_function(&runtime_library, "_evm_pop_u32").unwrap(),
            evm_push_function: find_runtime_function(&runtime_library, "_evm_push_u32").unwrap(),
            evm_burn_gas: find_runtime_function(&runtime_library, "_evm_burn_gas").unwrap(),
            evm_pc_function: find_runtime_function(&runtime_library, "_evm_set_pc").unwrap(),
            function_import_count: runtime_library.import_count(ImportCountType::Function),
            builder: parity_wasm::builder::from_module(runtime_library),
        }
    }

    /// Emit an empty `_start` function to make all WebAssembly runtimes happy.
    fn emit_wasm_start(self: &mut Compiler) {
        _ = self.emit_function(Some("_start".to_string()), vec![]);
    }

    /// Synthesizes a start function that initializes the EVM state with the
    /// correct configuration.
    fn emit_evm_start(self: &mut Compiler) {
        assert_ne!(self.evm_init_function, 0);

        self.evm_start_function = self.emit_function(
            Some("_evm_start".to_string()),
            vec![
                Instruction::I32Const(TABLE_OFFSET),
                Instruction::I64Const(self.config.chain_id.try_into().unwrap()), // --chain-id
                Instruction::I64Const(0),                                        // TODO: --balance
                Instruction::Call(self.evm_init_function),
            ],
        );
    }

    fn emit_abi_execute(self: &mut Compiler) {
        assert_ne!(self.evm_start_function, 0);
        assert_ne!(self.evm_exec_function, 0); // filled in during compile_cfg()

        _ = self.emit_function(
            Some("execute".to_string()),
            vec![
                Instruction::Call(self.evm_start_function),
                Instruction::Call(self.evm_exec_function),
                Instruction::I32Const(0),
                Instruction::I32Const(0), // output_types_len == 0 means no JSON encoding
                Instruction::Call(self.evm_post_exec_function),
            ],
        );
    }

    /// Synthesizes public wrapper methods for each function in the Solidity
    /// contract's ABI, enabling users to directly call a contract method
    /// without going through the low-level `execute` EVM dispatcher.
    pub fn emit_abi_methods(
        self: &mut Compiler,
        input_abi: Option<Functions>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert_ne!(self.evm_start_function, 0);
        assert_ne!(self.evm_call_function, 0);
        assert_ne!(self.evm_exec_function, 0); // filled in during compile_cfg()

        let mut data = Vec::with_capacity(self.abi_buffer_len);
        for func in input_abi.unwrap_or_default() {
            let names_off = data.len();
            for (i, input) in func.inputs.iter().enumerate() {
                if i > 0 {
                    write!(data, ",")?;
                }
                write!(data, "{}", input.name)?;
            }
            let names_len = data.len() - names_off;
            data.push(0); // NUL

            let types_off = data.len();
            for (i, input) in func.inputs.iter().enumerate() {
                if i > 0 {
                    write!(data, ",")?;
                }
                if abi_types::parse_param_type(&input.r#type).is_err() {
                    panic!("Unknown ABI type: {}", input.r#type);
                }
                write!(data, "{}", input.r#type)?;
            }
            let types_len = data.len() - types_off;
            data.push(0); // NUL

            let output_types_off = data.len();
            for (i, output) in func.outputs.iter().enumerate() {
                if i > 0 {
                    write!(data, ",")?;
                }
                if abi_types::parse_param_type(&output.r#type).is_err() {
                    panic!("Unknown ABI type: {}", output.r#type);
                }
                write!(data, "{}", output.r#type)?;
            }
            let output_types_len = data.len() - output_types_off;
            data.push(0); // NUL

            _ = self.emit_function(
                Some(func.name.clone()),
                vec![
                    Instruction::Call(self.evm_start_function),
                    Instruction::I32Const(func.selector() as i32),
                    Instruction::I32Const(names_off.try_into().unwrap()), // params_names_ptr
                    Instruction::I32Const(names_len.try_into().unwrap()), // params_names_len
                    Instruction::I32Const(types_off.try_into().unwrap()), // params_types_ptr
                    Instruction::I32Const(types_len.try_into().unwrap()), // params_types_len
                    Instruction::Call(self.evm_call_function),
                    Instruction::Call(self.evm_exec_function),
                    Instruction::I32Const(output_types_off.try_into().unwrap()), // output_types_off
                    Instruction::I32Const(output_types_len.try_into().unwrap()), // output_types_len
                    Instruction::Call(self.evm_post_exec_function),
                ],
            );
        }
        Ok(data)
    }

    //TODO self is only used for `evm_pop_function`
    fn unfold_cfg(
        &self,
        program: &Program,
        cfg_part: &ReSeq<SLabel<CaterpillarLabel<EvmLabel>>>,
        res: &mut Vec<Instruction>,
        wasm_idx2evm_idx: &mut HashMap<usize, usize>,
    ) {
        for block in cfg_part.0.iter() {
            match block {
                ReBlock::Block(inner_seq) => {
                    res.push(Instruction::Block(BlockType::NoResult)); //TODO block type?
                    self.unfold_cfg(program, inner_seq, res, wasm_idx2evm_idx);
                    res.push(Instruction::End);
                }
                ReBlock::Loop(inner_seq) => {
                    res.push(Instruction::Loop(BlockType::NoResult)); //TODO block type?
                    self.unfold_cfg(program, inner_seq, res, wasm_idx2evm_idx);
                    res.push(Instruction::End);
                }
                ReBlock::If(true_branch, false_branch) => {
                    res.push(Instruction::Call(self.evm_pop_function));
                    res.push(Instruction::If(BlockType::NoResult)); //TODO block type?
                    self.unfold_cfg(program, true_branch, res, wasm_idx2evm_idx);
                    res.push(Instruction::Else);
                    self.unfold_cfg(program, false_branch, res, wasm_idx2evm_idx);
                    res.push(Instruction::End);
                }
                ReBlock::Br(levels) => {
                    res.push(Instruction::Br((*levels).try_into().unwrap()));
                }
                ReBlock::Return => {
                    res.push(Instruction::Return);
                }
                ReBlock::Actions(block) => {
                    match block.origin {
                        CaterpillarLabel::Original(orig_label) => {
                            let block_code =
                                &program.0[orig_label.code_start.0..orig_label.code_end.0];
                            let block_len = orig_label.code_end.0 - orig_label.code_start.0;
                            let mut curr_idx = 0;
                            while curr_idx < block_len {
                                match &block_code[curr_idx..] {
                                    [p, j, ..] if p.is_push() && j.is_jump() => {
                                        // this is static jump, already accounted during cfg analysis. we only need to burn gas there
                                        let jump_gas = if j == &Opcode::JUMP { 8 } else { 10 };
                                        res.extend(vec![
                                            Instruction::I32Const(3),             // any push costs 3 gas
                                            Instruction::Call(self.evm_burn_gas), // burn it
                                            Instruction::I32Const(jump_gas),
                                            Instruction::Call(self.evm_burn_gas),
                                        ]);
                                        curr_idx += 2;
                                    }
                                    [j, ..] if j.is_jump() => {
                                        // this is dynamic jump
                                        let jump_gas = if j == &Opcode::JUMP { 8 } else { 10 };
                                        res.extend(vec![
                                            Instruction::Call(self.evm_pop_function),
                                            Instruction::SetLocal(0),
                                            Instruction::I32Const(jump_gas),
                                            Instruction::Call(self.evm_burn_gas),
                                        ]);
                                        curr_idx += 1;
                                    }
                                    [op, ..] => {
                                        wasm_idx2evm_idx
                                            .insert(res.len(), curr_idx + orig_label.code_start.0);
                                        if op.is_push() {
                                            let operands = encode_push(op);
                                            res.extend(operands);
                                        }
                                        let call = self.compile_operator(op);
                                        res.push(call);
                                        if op == &Opcode::RETURN {
                                            //TODO idk
                                            res.push(Instruction::Return);
                                        }
                                        curr_idx += 1;
                                    }
                                    [] => {}
                                }
                            }
                        }
                        CaterpillarLabel::Generated(a) => {
                            res.extend(vec![
                                Instruction::GetLocal(0),
                                Instruction::I32Const(a.label.0.try_into().unwrap()),
                                Instruction::I32Eq,
                                Instruction::Call(self.evm_push_function),
                            ]);
                        }
                    }
                }
            }
        }
    }

    // fn debug_only_exec_func(
    //     wasm: &[Instruction],
    //     evm_idx2offs: HashMap<usize, usize>,
    //     wasm_idx2evm_idx: HashMap<usize, usize>,
    // ) {
    //     let make_tab = |shift: &String, instr: &Instruction| -> String {
    //         let length = 80 - format!("{}{}", shift, instr).len();
    //         let mut res = "".to_string();
    //         for _ in 0..length {
    //             res.push(' ');
    //         }
    //         res
    //     };
    //     let mut shift = String::default();
    //     std::fs::write(
    //         "compiled.wat",
    //         wasm.iter()
    //             .enumerate()
    //             .map(|(idx, instr)| {
    //                 if instr == &Instruction::Else || instr == &Instruction::End {
    //                     for _ in 0..2 {
    //                         shift.pop();
    //                     }
    //                 }
    //                 let res = wasm_idx2evm_idx
    //                     .get(&idx)
    //                     .and_then(|x| evm_idx2offs.get(x))
    //                     .map_or_else(
    //                         || format!("{}{}", shift, instr,),
    //                         |offs| {
    //                             format!(
    //                                 "{}{} {}0x{:02x}",
    //                                 shift,
    //                                 instr,
    //                                 make_tab(&shift, instr),
    //                                 offs
    //                             )
    //                         },
    //                     );

    //                 match instr {
    //                     Instruction::Block(_)
    //                     | Instruction::Else
    //                     | Instruction::If(_)
    //                     | Instruction::Loop(_) => {
    //                         shift.push_str("  ");
    //                     }
    //                     _ => {}
    //                 }
    //                 res
    //             })
    //             .collect::<Vec<String>>()
    //             .join("\n"),
    //     )
    //     .expect("fs error");
    // }

    fn evm_wasm_dot_debug(
        program: &Program,
        basic_cfg: &BasicCfg,
        input_cfg: &ReSeq<SLabel<CaterpillarLabel<EvmLabel>>>,
        wasm: &[Instruction],
        wasm_idx2evm_idx: &HashMap<usize, usize>,
    ) {
        let evm_idx2offs = evm_idx_to_offs(program);

        let mut code_ranges: Vec<_> = basic_cfg.code_ranges.values().collect();
        code_ranges.sort_by_key(|x| x.start);

        let todo = code_ranges.iter().map(|range| {
            let start_end_str = format!("{}_{}", range.start, range.end);
            let range_nodes: Vec<String> = vec![];
            // let range_nodes: Vec<_> = range
            //     .map(|idx| {
            //         let op_offs = evm_idx2offs.get(&idx).unwrap();
            //         let e_op = program.0[idx];
            //         format!("evm_{}[label=\"0x{:x}: {}\"];", idx, op_offs, e_op)
            //     })
            //     .collect();
            format!(
                "subgraph cluster_evm_{} {{ label = \"{}\"
{}
}}",
                start_end_str,
                start_end_str,
                range_nodes.join("\n")
            )
        });

        let evm_nodes: Vec<_> = program
            .0
            .iter()
            .enumerate()
            .map(|(i, e_op)| {
                let op_offs = evm_idx2offs.get(&i).unwrap();
                format!("evm_{}[label=\"0x{:x}: {}\"];", i, op_offs, e_op)
            })
            .collect();

        let evm_links: Vec<_> = (0..program.0.len())
            .collect::<Vec<_>>()
            .windows(2) // TODO use `array_windows` (unstable for now)
            .map(|pair| {
                let a = pair[0];
                let b = pair[1];
                format!("evm_{a} -> evm_{b};")
            })
            .collect();

        let mut evm_lines = Vec::default();
        evm_lines.extend(evm_nodes);
        evm_lines.extend(evm_links);

        std::fs::write(
            "dbg.dot",
            format!(
                "digraph {{
subgraph cluster_evm {{ label = \"evm\"
{}
}}
}}",
                evm_lines.join("\n")
            ),
        )
        .expect("fs error");
    }

    /// Compiles the program's control-flow graph.
    fn compile_cfg(self: &mut Compiler, program: &Program) {
        assert_ne!(self.evm_start_function, 0); // filled in during emit_start()
        assert_eq!(self.evm_exec_function, 0); // filled in below

        let basic_cfg = basic_cfg(program);
        let relooped_cfg = relooped_cfg(&basic_cfg);

        let mut wasm: Vec<Instruction> = Default::default();
        let mut wasm_idx2evm_idx = Default::default();
        self.unfold_cfg(program, &relooped_cfg, &mut wasm, &mut wasm_idx2evm_idx);
        wasm.push(Instruction::End);

        Self::evm_wasm_dot_debug(program, &basic_cfg, &relooped_cfg, &wasm, &wasm_idx2evm_idx);

        let func_id = self.emit_function(Some("_evm_exec".to_string()), wasm);
        self.evm_exec_function = func_id;
    }

    /// Compiles the invocation of an EVM operator (operands must be already pushed).
    fn compile_operator(&self, op: &Opcode) -> Instruction {
        let op = op.zeroed();
        let op_idx = self.op_table.get(&op).unwrap();
        Instruction::Call(*op_idx)
    }

    fn emit_function(&mut self, name: Option<String>, mut code: Vec<Instruction>) -> FunctionIndex {
        match code.last() {
            Some(Instruction::End) => {}
            Some(_) | None => code.push(Instruction::End),
        };

        let func_sig = SignatureBuilder::new()
            .with_params(vec![])
            .with_results(vec![])
            .build_sig();

        let func_locals = vec![Local::new(1, ValueType::I32)]; // needed for dynamic branches
        let func_body = FuncBody::new(func_locals, Instructions::new(code));

        let func = FunctionBuilder::new()
            .with_signature(func_sig)
            .with_body(func_body)
            .build();

        let func_loc = self.builder.push_function(func);

        let func_idx = func_loc.signature + self.function_import_count as u32; // TODO: https://github.com/paritytech/parity-wasm/issues/304

        if let Some(name) = name {
            let func_export = ExportEntry::new(name, Internal::Function(func_idx));

            let _ = self.builder.push_export(func_export);
        }

        func_idx
    }
}

fn make_op_table(module: &Module) -> HashMap<Opcode, FunctionIndex> {
    let mut result: HashMap<Opcode, FunctionIndex> = HashMap::new();
    for export in module.export_section().unwrap().entries() {
        match export.internal() {
            &Internal::Function(op_idx) => match export.field() {
                "_abi_buffer" | "_evm_start" | "_evm_init" | "_evm_call" | "_evm_exec"
                | "_evm_post_exec" | "_evm_pop_u32" | "_evm_push_u32" | "_evm_burn_gas"
                | "_evm_set_pc" | "execute" => {}
                export_sym => match parse_opcode(&export_sym.to_ascii_uppercase()) {
                    None => unreachable!(), // TODO
                    Some(op) => _ = result.insert(op, op_idx),
                },
            },
            _ => continue,
        }
    }
    result
}

fn find_runtime_function(module: &Module, name: &str) -> Option<FunctionIndex> {
    for export in module.export_section().unwrap().entries() {
        match export.internal() {
            &Internal::Function(op_idx) => {
                if export.field() == name {
                    return Some(op_idx);
                }
            }
            _ => continue,
        }
    }
    None // not found
}

fn find_runtime_function_type(module: &Module) -> Option<TypeIndex> {
    for (type_id, r#type) in module.type_section().unwrap().types().iter().enumerate() {
        match r#type {
            Type::Function(function_type) => {
                if function_type.params().is_empty() && function_type.results().is_empty() {
                    return Some(type_id.try_into().unwrap());
                }
            }
        }
    }
    None // not found
}

fn find_abi_buffer(module: &Module) -> Option<DataOffset> {
    for export in module.export_section().unwrap().entries() {
        match export.internal() {
            &Internal::Global(idx) => {
                if export.field() == "_abi_buffer" {
                    // found it
                    let global = module
                        .global_section()
                        .unwrap()
                        .entries()
                        .get(idx as usize)
                        .unwrap();
                    match global.init_expr().code().first().unwrap() {
                        Instruction::I32Const(off) => return Some(*off),
                        _ => return None,
                    }
                }
            }
            _ => continue,
        }
    }
    None // not found
}
