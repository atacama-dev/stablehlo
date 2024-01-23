#[derive(Debug, Clone, Copy)]
pub enum ElementKind {
    Operand,
    Result,
}

impl ElementKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Operand => "operand",
            Self::Result => "result",
        }
    }
}
