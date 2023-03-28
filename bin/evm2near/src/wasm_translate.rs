// grabbed from https://github.com/bytecodealliance/wasm-tools/blob/main/crates/wasm-mutate/src/mutators/translate.rs

use anyhow::{Error, Result};
use wasm_encoder::*;
use wasmparser::{DataKind, ElementKind, FunctionBody, Global, Operator, Type};

pub type Params = Vec<ValType>;
pub type Results = Vec<ValType>;

pub struct Import {
    module: String,
    field: String,
    ty: EntityType,
}

pub type TypeIndex = u32;

pub struct Glob {
    //todo rename
    typ: GlobalType,
    init: ConstExpr,
}

pub struct Export {
    name: String,
    kind: ExportKind,
    index: u32,
}

pub struct ModuleBuilder<'a> {
    types: Vec<(Params, Results)>,
    imports: Vec<Import>,
    functions: Vec<TypeIndex>,
    tables: Vec<TableType>,
    memories: Vec<MemoryType>,
    globals: Vec<Glob>,
    exports: Vec<Export>,
    start_sect: StartSection,
    elements: Vec<ElementSegment<'a>>,
    code: Vec<Function>,
    data: Vec<DataSegment<'a, Vec<u8>>>,
}

impl ModuleBuilder<'_> {
    fn new() -> Self {
        ModuleBuilder {
            types: Default::default(),
            imports: Default::default(),
            functions: Default::default(),
            tables: Default::default(),
            memories: Default::default(),
            globals: Default::default(),
            exports: Default::default(),
            start_sect: StartSection { function_index: 0 },
            elements: Default::default(),
            code: Default::default(),
            data: Default::default(),
        }
    }

    pub fn build(self) -> Module {
        let mut m = Module::new();
        let mut type_section = TypeSection::new();
        for (params, results) in self.types {
            type_section.function(params, results);
        }
        m.section(&type_section);

        let mut import_section = ImportSection::new();
        for i in self.imports {
            import_section.import(&i.module, &i.field, i.ty);
        }
        m.section(&import_section);

        let mut function_section = FunctionSection::new();
        for f in self.functions {
            function_section.function(f);
        }
        m.section(&function_section);

        let mut table_section = TableSection::new();
        for t in self.tables {
            table_section.table(t);
        }
        m.section(&table_section);

        let mut memory_section = MemorySection::new();
        for m in self.memories {
            memory_section.memory(m);
        }
        m.section(&memory_section);

        let mut global_section = GlobalSection::new();
        for global in self.globals {
            global_section.global(global.typ, &global.init);
        }
        m.section(&global_section);

        let mut export_section = ExportSection::new();
        for e in self.exports {
            export_section.export(&e.name, e.kind, e.index);
        }
        m.section(&export_section);

        m.section(&self.start_sect);

        let mut element_section = ElementSection::new();
        for e in self.elements {
            element_section.segment(e);
        }
        m.section(&element_section);

        let mut code_section = CodeSection::new();
        for c in self.code {
            code_section.function(&c);
        }
        m.section(&code_section);

        //todo data count section

        let mut data_section = DataSection::new();
        for d in self.data {
            data_section.segment(d);
        }
        m.section(&data_section);

        m
    }
}

pub fn parse(wasm: Vec<u8>) -> Result<ModuleBuilder<'static>> {
    let parsed = wasmparser::Parser::new(0)
        .parse_all(wasm.as_slice())
        .map(|p| p.unwrap())
        .collect::<Vec<_>>();

    let translator = Translator;
    let t_b = Box::new(translator);
    let t = Box::leak(t_b); //todo remove that garbage

    let mut code_section_size: Option<u32> = None;

    let mut builder = ModuleBuilder::new();
    for p in parsed {
        match p {
            wasmparser::Payload::Version {
                num: _version,
                encoding,
                range: _range,
            } => {
                assert_eq!(encoding, wasmparser::Encoding::Module);
            }
            wasmparser::Payload::TypeSection(type_section) => {
                for typ in type_section {
                    let (params, results) = t.type_def(typ?)?;
                    builder.types.push((params, results));
                }
            }
            wasmparser::Payload::ImportSection(import_section) => {
                for import in import_section {
                    let import = import?;
                    let typ = t.type_ref(import.ty)?;
                    builder.imports.push(Import {
                        module: import.module.to_string(),
                        field: import.name.to_string(),
                        ty: typ,
                    });
                }
            }
            wasmparser::Payload::FunctionSection(function_section) => {
                for function in function_section {
                    builder.functions.push(function?);
                }
            }
            wasmparser::Payload::TableSection(table_section) => {
                for table in table_section {
                    let table = table?;
                    let table_type = t.table_type(&table.ty)?; // todo TableInit is not used!
                    builder.tables.push(table_type);
                }
            }
            wasmparser::Payload::MemorySection(memory_section) => {
                for mem in memory_section {
                    let mem = mem?;
                    let memory_type = t.memory_type(&mem)?;
                    builder.memories.push(memory_type);
                }
            }
            wasmparser::Payload::TagSection(tag_section) => {
                for tag in tag_section {
                    let tag = tag?;
                    let _tag_type = t.tag_type(&tag)?;
                    panic!("unspecified section (spec doesnt have that one! WTF)");
                }
            }
            wasmparser::Payload::GlobalSection(global_section) => {
                for glob in global_section {
                    let (typ, init) = t.global(glob?)?;
                    builder.globals.push(Glob { typ, init });
                }
            }
            wasmparser::Payload::ExportSection(export_section) => {
                for export in export_section {
                    let export = export?;
                    builder.exports.push(Export {
                        name: export.name.to_string(),
                        kind: t.ext_kind(export.kind),
                        index: export.index,
                    });
                }
            }
            wasmparser::Payload::StartSection {
                func: function_index,
                range: _range,
            } => {
                let start_sect = wasm_encoder::StartSection { function_index };
                builder.start_sect = start_sect;
            }
            wasmparser::Payload::ElementSection(element_section) => {
                for element in element_section {
                    let element_segment = t.element(element?)?;
                    builder.elements.push(element_segment);
                }
            }
            wasmparser::Payload::DataCountSection { count: _, range: _ } => {
                panic!("unspecified section (spec doesnt have that one! WTF)");
            }
            wasmparser::Payload::DataSection(data_section) => {
                for data in data_section {
                    let data_seg = t.data(data?)?;
                    builder.data.push(data_seg);
                }
            }
            wasmparser::Payload::CodeSectionStart {
                count,
                range: _,
                size: _,
            } => {
                code_section_size = Some(count);
            }
            wasmparser::Payload::CodeSectionEntry(code_section_entry) => {
                assert!(code_section_size.is_some());
                let code_seg = t.code(code_section_entry)?;
                builder.code.push(code_seg);
            }
            wasmparser::Payload::ModuleSection {
                parser: _,
                range: _,
            } => todo!(),
            wasmparser::Payload::InstanceSection(_) => todo!(),
            wasmparser::Payload::CoreTypeSection(_) => todo!(),
            wasmparser::Payload::ComponentSection {
                parser: _,
                range: _,
            } => todo!(),
            wasmparser::Payload::ComponentInstanceSection(_) => todo!(),
            wasmparser::Payload::ComponentAliasSection(_) => todo!(),
            wasmparser::Payload::ComponentTypeSection(_) => todo!(),
            wasmparser::Payload::ComponentCanonicalSection(_) => todo!(),
            wasmparser::Payload::ComponentStartSection { start: _, range: _ } => todo!(),
            wasmparser::Payload::ComponentImportSection(_) => todo!(),
            wasmparser::Payload::ComponentExportSection(_) => todo!(),
            wasmparser::Payload::CustomSection(_) => todo!(),
            wasmparser::Payload::UnknownSection {
                id: _,
                contents: _,
                range: _,
            } => todo!(),
            wasmparser::Payload::End(_end) => {}
        }
    }

    assert_eq!(
        u32::try_from(builder.code.len()).unwrap(),
        code_section_size.unwrap()
    );

    Ok(builder)
}

#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub enum ConstExprKind {
    Global,
    ElementOffset,
    ElementFunction,
    DataOffset,
    TableInit,
}

pub struct Translator;

impl Translator {
    fn type_ref(&self, type_ref: wasmparser::TypeRef) -> Result<wasm_encoder::EntityType> {
        match type_ref {
            wasmparser::TypeRef::Func(f) => Ok(EntityType::Function(f)),
            wasmparser::TypeRef::Table(t) => {
                let element_type = self.refty(&t.element_type)?;
                Ok(EntityType::Table(TableType {
                    element_type,
                    minimum: t.initial,
                    maximum: t.maximum,
                }))
            }
            wasmparser::TypeRef::Memory(m) => Ok(EntityType::Memory(MemoryType {
                memory64: m.memory64,
                minimum: m.initial,
                maximum: m.maximum,
                shared: m.shared,
            })),
            wasmparser::TypeRef::Global(g) => Ok(EntityType::Global(GlobalType {
                val_type: self.ty(&g.content_type)?,
                mutable: g.mutable,
            })),
            wasmparser::TypeRef::Tag(t) => Ok(EntityType::Tag(self.tag_type(&t)?)),
        }
    }

    pub fn type_def(&self, ty: Type) -> Result<(Vec<ValType>, Vec<ValType>)> {
        match ty {
            Type::Func(f) => {
                let params = f
                    .params()
                    .iter()
                    .map(|ty| self.ty(ty))
                    .collect::<Result<Vec<_>>>()?;
                let results = f
                    .results()
                    .iter()
                    .map(|ty| self.ty(ty))
                    .collect::<Result<Vec<_>>>()?;
                Ok((params, results))
            }
        }
    }

    pub fn ext_kind(&self, ekind: wasmparser::ExternalKind) -> ExportKind {
        match ekind {
            wasmparser::ExternalKind::Func => ExportKind::Func,
            wasmparser::ExternalKind::Table => ExportKind::Table,
            wasmparser::ExternalKind::Memory => ExportKind::Memory,
            wasmparser::ExternalKind::Global => ExportKind::Global,
            wasmparser::ExternalKind::Tag => ExportKind::Tag,
        }
    }

    pub fn table_type(&self, ty: &wasmparser::TableType) -> Result<wasm_encoder::TableType> {
        Ok(wasm_encoder::TableType {
            element_type: self.refty(&ty.element_type)?,
            minimum: ty.initial,
            maximum: ty.maximum,
        })
    }

    pub fn memory_type(&self, ty: &wasmparser::MemoryType) -> Result<wasm_encoder::MemoryType> {
        Ok(wasm_encoder::MemoryType {
            memory64: ty.memory64,
            minimum: ty.initial,
            maximum: ty.maximum,
            shared: ty.shared,
        })
    }

    pub fn global_type(&self, ty: &wasmparser::GlobalType) -> Result<wasm_encoder::GlobalType> {
        Ok(wasm_encoder::GlobalType {
            val_type: self.ty(&ty.content_type)?,
            mutable: ty.mutable,
        })
    }

    pub fn tag_type(&self, ty: &wasmparser::TagType) -> Result<wasm_encoder::TagType> {
        Ok(wasm_encoder::TagType {
            kind: TagKind::Exception,
            func_type_idx: ty.func_type_idx,
        })
    }

    pub fn ty(&self, ty: &wasmparser::ValType) -> Result<ValType> {
        match ty {
            wasmparser::ValType::I32 => Ok(ValType::I32),
            wasmparser::ValType::I64 => Ok(ValType::I64),
            wasmparser::ValType::F32 => Ok(ValType::F32),
            wasmparser::ValType::F64 => Ok(ValType::F64),
            wasmparser::ValType::V128 => Ok(ValType::V128),
            wasmparser::ValType::Ref(ty) => Ok(ValType::Ref(self.refty(ty)?)),
        }
    }

    pub fn refty(&self, ty: &wasmparser::RefType) -> Result<RefType> {
        Ok(RefType {
            nullable: ty.nullable,
            heap_type: self.heapty(&ty.heap_type)?,
        })
    }

    pub fn heapty(&self, ty: &wasmparser::HeapType) -> Result<HeapType> {
        match ty {
            wasmparser::HeapType::Func => Ok(HeapType::Func),
            wasmparser::HeapType::Extern => Ok(HeapType::Extern),
            wasmparser::HeapType::TypedFunc(i) => Ok(HeapType::TypedFunc((*i).into())),
        }
    }

    pub fn global(&self, global: Global) -> Result<(GlobalType, ConstExpr)> {
        let ty = self.global_type(&global.ty)?;
        let insn = self.const_expr(&global.init_expr, ConstExprKind::Global)?;
        Ok((ty, insn))
    }

    pub fn const_expr(
        &self,
        e: &wasmparser::ConstExpr<'_>,
        ctx: ConstExprKind,
    ) -> Result<wasm_encoder::ConstExpr> {
        let mut e = e.get_operators_reader();
        let mut offset_bytes = Vec::new();
        let op = e.read()?;
        if let ConstExprKind::ElementFunction = ctx {
            match op {
                Operator::RefFunc { .. }
                | Operator::RefNull {
                    hty: wasmparser::HeapType::Func,
                    ..
                }
                | Operator::GlobalGet { .. } => {}
                _ => return Err(Error::msg("no mutations applicable")),
            }
        }
        self.op(&op)?.encode(&mut offset_bytes);
        match e.read()? {
            Operator::End if e.eof() => {}
            _ => return Err(Error::msg("no mutations applicable")),
        }
        Ok(wasm_encoder::ConstExpr::raw(offset_bytes))
    }

    pub fn element(&self, element: wasmparser::Element<'_>) -> Result<ElementSegment> {
        let mode: ElementMode<'static> = match &element.kind {
            ElementKind::Active {
                table_index,
                offset_expr,
            } => {
                let offset = self.const_expr(offset_expr, ConstExprKind::ElementOffset)?;
                let offset_box = Box::new(offset);
                ElementMode::Active {
                    table: Some(*table_index),
                    offset: Box::leak(offset_box),
                }
            }
            ElementKind::Passive => ElementMode::Passive,
            ElementKind::Declared => ElementMode::Declared,
        };
        let element_type = self.refty(&element.ty)?;
        let elements = match element.items {
            wasmparser::ElementItems::Functions(reader) => {
                let functions = reader.into_iter().collect::<Result<Vec<_>, _>>()?;
                let functions_box = Box::new(functions);
                Elements::Functions(Box::leak(functions_box))
            }
            wasmparser::ElementItems::Expressions(reader) => {
                let exprs = reader
                    .into_iter()
                    .map(|f| self.const_expr(&f?, ConstExprKind::ElementFunction))
                    .collect::<Result<Vec<_>, _>>()?;
                let exprs_box = Box::new(exprs);
                Elements::Expressions(Box::leak(exprs_box))
            }
        };
        Ok(ElementSegment {
            mode,
            element_type,
            elements,
        })
    }

    #[allow(unused_variables)]
    pub fn op(&self, op: &Operator<'_>) -> Result<Instruction<'static>> {
        use wasm_encoder::Instruction as I;

        macro_rules! translate {
            ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident)*) => {
                Ok(match op {
                    $(
                        wasmparser::Operator::$op $({ $($arg),* })? => {
                            $(
                                $(let $arg = translate!(map $arg $arg);)*
                            )?
                            translate!(build $op $($($arg)*)?)
                        }
                    )*
                })
            };

            // This case is used to map, based on the name of the field, from the
            // wasmparser payload type to the wasm-encoder payload type through
            // `Translator` as applicable.
            (map $arg:ident tag_index) => (*$arg);
            (map $arg:ident function_index) => (*$arg);
            (map $arg:ident table) => (*$arg);
            (map $arg:ident table_index) => (*$arg);
            (map $arg:ident table) => (*$arg);
            (map $arg:ident dst_table) => (*$arg);
            (map $arg:ident src_table) => (*$arg);
            (map $arg:ident type_index) => (*$arg);
            (map $arg:ident global_index) => (*$arg);
            (map $arg:ident mem) => (*$arg);
            (map $arg:ident src_mem) => (*$arg);
            (map $arg:ident dst_mem) => (*$arg);
            (map $arg:ident data_index) => (*$arg);
            (map $arg:ident elem_index) => (*$arg);
            (map $arg:ident blockty) => (self.block_type($arg)?);
            (map $arg:ident relative_depth) => (*$arg);
            (map $arg:ident targets) => ((
                $arg
                    .targets()
                    .collect::<Result<Vec<_>, wasmparser::BinaryReaderError>>()?
                    .into(),
                $arg.default(),
            ));
            (map $arg:ident table_byte) => (());
            (map $arg:ident mem_byte) => (());
            (map $arg:ident flags) => (());
            (map $arg:ident ty) => (self.ty($arg)?);
            (map $arg:ident hty) => (self.heapty($arg)?);
            (map $arg:ident memarg) => (self.memarg($arg)?);
            (map $arg:ident local_index) => (*$arg);
            (map $arg:ident value) => ($arg);
            (map $arg:ident lane) => (*$arg);
            (map $arg:ident lanes) => (*$arg);

            // This case takes the arguments of a wasmparser instruction and creates
            // a wasm-encoder instruction. There are a few special cases for where
            // the structure of a wasmparser instruction differs from that of
            // wasm-encoder.
            (build $op:ident) => (I::$op);
            (build BrTable $arg:ident) => (I::BrTable($arg.0, $arg.1));
            (build I32Const $arg:ident) => (I::I32Const(*$arg));
            (build I64Const $arg:ident) => (I::I64Const(*$arg));
            (build F32Const $arg:ident) => (I::F32Const(f32::from_bits($arg.bits())));
            (build F64Const $arg:ident) => (I::F64Const(f64::from_bits($arg.bits())));
            (build V128Const $arg:ident) => (I::V128Const($arg.i128()));
            (build $op:ident $arg:ident) => (I::$op($arg));
            (build CallIndirect $ty:ident $table:ident $_:ident) => (I::CallIndirect {
                ty: $ty,
                table: $table,
            });
            (build ReturnCallIndirect $ty:ident $table:ident) => (I::ReturnCallIndirect {
                ty: $ty,
                table: $table,
            });
            (build MemoryGrow $mem:ident $_:ident) => (I::MemoryGrow($mem));
            (build MemorySize $mem:ident $_:ident) => (I::MemorySize($mem));
            (build $op:ident $($arg:ident)*) => (I::$op { $($arg),* });
        }

        wasmparser::for_each_operator!(translate)
    }

    pub fn block_type(&self, ty: &wasmparser::BlockType) -> Result<BlockType> {
        match ty {
            wasmparser::BlockType::Empty => Ok(BlockType::Empty),
            wasmparser::BlockType::Type(ty) => Ok(BlockType::Result(self.ty(ty)?)),
            wasmparser::BlockType::FuncType(f) => Ok(BlockType::FunctionType(*f)),
        }
    }

    pub fn memarg(&self, memarg: &wasmparser::MemArg) -> Result<MemArg> {
        Ok(MemArg {
            offset: memarg.offset,
            align: memarg.align.into(),
            memory_index: memarg.memory,
        })
    }

    pub fn data(&self, data: wasmparser::Data<'_>) -> Result<DataSegment<Vec<u8>>> {
        let mode: DataSegmentMode<'static> = match &data.kind {
            DataKind::Active {
                memory_index,
                offset_expr,
            } => {
                let offset = self.const_expr(offset_expr, ConstExprKind::DataOffset)?;
                let offset_box = Box::new(offset);
                DataSegmentMode::Active {
                    memory_index: *memory_index,
                    offset: Box::leak(offset_box), // todo satisfy static lifetime for the `mode` binding
                }
            }
            DataKind::Passive => DataSegmentMode::Passive,
        };
        let data = data.data.to_vec();
        Ok(DataSegment { mode, data })
    }

    pub fn code(&self, body: FunctionBody<'_>) -> Result<Function> {
        let locals = body
            .get_locals_reader()?
            .into_iter()
            .map(|local| {
                let (cnt, ty) = local?;
                Ok((cnt, self.ty(&ty)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut func = Function::new(locals);

        let mut reader = body.get_operators_reader()?;
        reader.allow_memarg64(true);
        for op in reader {
            let op = op?;
            func.instruction(&self.op(&op)?);
        }
        Ok(func)
    }
}
