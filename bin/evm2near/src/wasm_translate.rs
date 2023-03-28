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
    globals: Vec<(GlobalType, ConstExpr)>,
    exports: Vec<Export>,
    start_sect: StartSection,
    elements: Vec<ElementSegment<'a>>,
    code: Vec<Function>,
    data: Vec<()>,
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

    fn build(self) -> Module {
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
        for (global_type, init_expr) in self.globals {
            global_section.global(global_type, &init_expr);
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

        m
    }
}

pub fn parse(wasm: Vec<u8>) -> Result<Module> {
    let parsed = wasmparser::Parser::new(0)
        .parse_all(wasm.as_slice())
        .map(|p| p.unwrap())
        .collect::<Vec<_>>();

    let mut t = Translator;

    let mut code_section_size: Option<u32> = None;
    let mut code_section = wasm_encoder::CodeSection::new();

    let mut builder = ModuleBuilder::new();
    let mut m = wasm_encoder::Module::new();
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
                let mut type_sect = wasm_encoder::TypeSection::new();
                for typ in type_section {
                    let (params, results) = t.type_def(typ?)?;
                    type_sect.function(params, results);
                }
                m.section(&type_sect);
            }
            wasmparser::Payload::ImportSection(import_section) => {
                let mut import_sect = wasm_encoder::ImportSection::new();
                for import in import_section {
                    let import = import?;
                    let typ = t.type_ref(import.ty)?;
                    import_sect.import(import.module, import.name, typ);
                }
                m.section(&import_sect);
            }
            wasmparser::Payload::FunctionSection(function_section) => {
                let mut function_sect = wasm_encoder::FunctionSection::new();
                for function in function_section {
                    function_sect.function(function?);
                }
                m.section(&function_sect);
            }
            wasmparser::Payload::TableSection(table_section) => {
                let mut table_sect = wasm_encoder::TableSection::new();
                for table in table_section {
                    let table = table?;
                    let table_type = t.table_type(&table.ty)?; // todo TableInit is not used!
                    table_sect.table(table_type);
                }
                m.section(&table_sect);
            }
            wasmparser::Payload::MemorySection(memory_section) => {
                let mut memory_sect = wasm_encoder::MemorySection::new();
                for mem in memory_section {
                    let mem = mem?;
                    let memory_type = t.memory_type(&mem)?;
                    memory_sect.memory(memory_type);
                }
                m.section(&memory_sect);
            }
            wasmparser::Payload::TagSection(tag_section) => {
                let mut tag_sect = wasm_encoder::TagSection::new();
                for tag in tag_section {
                    let tag = tag?;
                    let tag_type = t.tag_type(&tag)?;
                    tag_sect.tag(tag_type);
                }
                m.section(&tag_sect);
            }
            wasmparser::Payload::GlobalSection(global_section) => {
                let mut glob_sect = wasm_encoder::GlobalSection::new();
                for glob in global_section {
                    let (global, init) = t.global(glob?)?;
                    glob_sect.global(global, &init);
                }
                m.section(&glob_sect);
            }
            wasmparser::Payload::ExportSection(export_section) => {
                let mut export_sect = wasm_encoder::ExportSection::new();
                for export in export_section {
                    let export = export?;
                    export_sect.export(export.name, t.ext_kind(export.kind), export.index);
                }
                m.section(&export_sect);
            }
            wasmparser::Payload::StartSection {
                func: function_index,
                range: _range,
            } => {
                let start_sect = wasm_encoder::StartSection { function_index };
                m.section(&start_sect);
            }
            wasmparser::Payload::ElementSection(element_section) => {
                let mut element_sect = wasm_encoder::ElementSection::new();
                for element in element_section {
                    let element_segment = t.element(element?)?;
                    element_sect.segment(element_segment);
                }
                m.section(&element_sect);
            }
            wasmparser::Payload::DataCountSection { count, range: _ } => {
                let data_count_sect = wasm_encoder::DataCountSection { count };
                m.section(&data_count_sect);
            }
            wasmparser::Payload::DataSection(data_section) => {
                let mut data_sect = wasm_encoder::DataSection::new();
                for data in data_section {
                    let data_seg = t.data(data?)?;
                    data_sect.segment(data_seg);
                }
                m.section(&data_sect);
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
                code_section.function(&code_seg);
                let desired_size = code_section_size.unwrap();
                if code_section.len() == desired_size {
                    m.section(&code_section);
                    code_section_size = None;
                }
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

    Ok(m)
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
    fn type_ref(&mut self, type_ref: wasmparser::TypeRef) -> Result<wasm_encoder::EntityType> {
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

    pub fn type_def(&mut self, ty: Type) -> Result<(Vec<ValType>, Vec<ValType>)> {
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

    pub fn ext_kind(&mut self, ekind: wasmparser::ExternalKind) -> ExportKind {
        match ekind {
            wasmparser::ExternalKind::Func => ExportKind::Func,
            wasmparser::ExternalKind::Table => ExportKind::Table,
            wasmparser::ExternalKind::Memory => ExportKind::Memory,
            wasmparser::ExternalKind::Global => ExportKind::Global,
            wasmparser::ExternalKind::Tag => ExportKind::Tag,
        }
    }

    pub fn table_type(&mut self, ty: &wasmparser::TableType) -> Result<wasm_encoder::TableType> {
        Ok(wasm_encoder::TableType {
            element_type: self.refty(&ty.element_type)?,
            minimum: ty.initial,
            maximum: ty.maximum,
        })
    }

    pub fn memory_type(&mut self, ty: &wasmparser::MemoryType) -> Result<wasm_encoder::MemoryType> {
        Ok(wasm_encoder::MemoryType {
            memory64: ty.memory64,
            minimum: ty.initial,
            maximum: ty.maximum,
            shared: ty.shared,
        })
    }

    pub fn global_type(&mut self, ty: &wasmparser::GlobalType) -> Result<wasm_encoder::GlobalType> {
        Ok(wasm_encoder::GlobalType {
            val_type: self.ty(&ty.content_type)?,
            mutable: ty.mutable,
        })
    }

    pub fn tag_type(&mut self, ty: &wasmparser::TagType) -> Result<wasm_encoder::TagType> {
        Ok(wasm_encoder::TagType {
            kind: TagKind::Exception,
            func_type_idx: ty.func_type_idx,
        })
    }

    pub fn ty(&mut self, ty: &wasmparser::ValType) -> Result<ValType> {
        match ty {
            wasmparser::ValType::I32 => Ok(ValType::I32),
            wasmparser::ValType::I64 => Ok(ValType::I64),
            wasmparser::ValType::F32 => Ok(ValType::F32),
            wasmparser::ValType::F64 => Ok(ValType::F64),
            wasmparser::ValType::V128 => Ok(ValType::V128),
            wasmparser::ValType::Ref(ty) => Ok(ValType::Ref(self.refty(ty)?)),
        }
    }

    pub fn refty(&mut self, ty: &wasmparser::RefType) -> Result<RefType> {
        Ok(RefType {
            nullable: ty.nullable,
            heap_type: self.heapty(&ty.heap_type)?,
        })
    }

    pub fn heapty(&mut self, ty: &wasmparser::HeapType) -> Result<HeapType> {
        match ty {
            wasmparser::HeapType::Func => Ok(HeapType::Func),
            wasmparser::HeapType::Extern => Ok(HeapType::Extern),
            wasmparser::HeapType::TypedFunc(i) => Ok(HeapType::TypedFunc((*i).into())),
        }
    }

    pub fn global(&mut self, global: Global) -> Result<(GlobalType, ConstExpr)> {
        let ty = self.global_type(&global.ty)?;
        let insn = self.const_expr(&global.init_expr, ConstExprKind::Global)?;
        Ok((ty, insn))
    }

    pub fn const_expr(
        &mut self,
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

    pub fn element(&mut self, element: wasmparser::Element<'_>) -> Result<ElementSegment> {
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
    pub fn op(&mut self, op: &Operator<'_>) -> Result<Instruction<'static>> {
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

    pub fn block_type(&mut self, ty: &wasmparser::BlockType) -> Result<BlockType> {
        match ty {
            wasmparser::BlockType::Empty => Ok(BlockType::Empty),
            wasmparser::BlockType::Type(ty) => Ok(BlockType::Result(self.ty(ty)?)),
            wasmparser::BlockType::FuncType(f) => Ok(BlockType::FunctionType(*f)),
        }
    }

    pub fn memarg(&mut self, memarg: &wasmparser::MemArg) -> Result<MemArg> {
        Ok(MemArg {
            offset: memarg.offset,
            align: memarg.align.into(),
            memory_index: memarg.memory,
        })
    }

    pub fn data(&mut self, data: wasmparser::Data<'_>) -> Result<DataSegment<Vec<u8>>> {
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

    pub fn code(&mut self, body: FunctionBody<'_>) -> Result<Function> {
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
