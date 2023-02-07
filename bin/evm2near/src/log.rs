/// gets iter to collection of elements with implemented Display and prints it in file
/// each element on separate string
pub fn log<Iter : Iterator, Stringable : std::fmt::Display>(iter : Iter, file : Stringable) 
where <Iter as Iterator>::Item: std::fmt::Display {
    std::fs::write(file.to_string(), iter.map(|x| x.to_string()).collect::<Vec<_>>().join("\n")).expect("fs error");
}

// same but as macros
#[macro_export]
macro_rules! log {
    ($iter:expr, $file:expr) => {
        std::fs::write($file.to_string(), $iter.map(|x| x.to_string()).collect::<Vec<_>>().join("\n")).expect("fs error");
    };
}
