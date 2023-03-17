use csv::Writer;
use serde::{Deserialize};
use serde_json::{json, Value};
use std::{fs::File, ffi::OsString};
use std::process::Command;


#[derive(Debug, Deserialize)]
struct Input {
    pub method: Box<str>,
    pub input: Value,
}

const TERA: u64 = 1000000000000_u64;

async fn bench_contract(wtr: &mut Writer<File>, name_os: OsString, commit: String) -> anyhow::Result<()> {
    let name = &name_os.to_str().unwrap()[0..name_os.len() - 5];
    println!("Name = {}", name);
    let worker = near_workspaces::sandbox().await?;
    let wasm = std::fs::read(format!("{}.wasm", name))?;
    let contract = worker.dev_deploy(&wasm).await?;

    let inputs: Vec<Input> = serde_json::from_str(
        &std::fs::read_to_string(format!("inputs/{}.json", name))
            .expect("Unable to read file"),
    )
    .expect("JSON does not have correct format.");
    let deposit = 10000000000000000000000_u128;
    for input in &inputs {
        let outcome = contract
            .call(&input.method)
            .args_json(json!(input.input))
            .deposit(deposit)
            .gas(near_units::parse_gas!("300 TGas") as u64)
            .transact()
            .await?;
        for failure in &outcome.failures() {
            println!("{:#?}", failure);
        }
        assert!(outcome.is_success());
        wtr.write_record(&[
            name.to_string(),
            input.method.to_string(),
            outcome.outcome().gas_burnt.to_string(),
            outcome.total_gas_burnt.to_string(),
            (outcome.outcome().gas_burnt / TERA).to_string(),
            (outcome.total_gas_burnt / TERA).to_string(),
            input.input.to_string(),
            commit.clone(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {

    let paths = std::fs::read_dir("inputs/").unwrap();

    let contracts = paths.into_iter().map(|dir| dir.unwrap().file_name()).collect::<Vec<_>>();



    let output = Command::new("sh")
                .arg("-c")
                .arg("git rev-parse --short HEAD")
                .output()
                .expect("failed to execute process");
    
    let stdout = output.stdout;
    let mut commit = std::str::from_utf8(&stdout).unwrap().to_string();
    commit.pop();  // to remove \n in the end

    

    let mut wtr = Writer::from_path(format!("{}.csv", commit).to_string())?;
    wtr.write_record([
        "Contract",
        "Method",
        "Gas burned",
        "Gas used",
        "Tgas burned",
        "Tgas used",
        "Input",
        "Commit",
    ])?;

    for contract in contracts {
        bench_contract(&mut wtr, contract, commit.clone()).await?;
    }

    wtr.flush()?;
    Ok(())
}