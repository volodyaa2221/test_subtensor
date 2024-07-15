mod math;
use std::fs::File;
use std::io::{self, Read};
use math::*;
use regex::Regex;
use substrate_fixed::types::I32F32;

const DEBUG_UID: usize = 142;

fn main() {

    let active_stake = parse_incentive_from_file("debugging_logs/active_stake.txt");    
    println!("Active Stake Total Sum:{:?}", active_stake.iter().sum::<I32F32>());

    let mut stake_idx_list = Vec::new();
    for (index, stake) in active_stake.iter().enumerate() {
        if  *stake > I32F32::from_num(0) {
            stake_idx_list.push(index);
        }
    }
    println!("-----------------Weights-------------");    
    let mut weights = parse_weights_from_file("debugging_logs/weight_6_final.txt");    

    let weight_delta: f32 = 1_f32/ (stake_idx_list.len() as f32);
    for index in stake_idx_list {
        weights[index].push((DEBUG_UID as u16, I32F32::from_num(weight_delta)));
    }

    let mut sum_weights: Vec<I32F32> = vec![I32F32::from_num(0.0); weights.len()];
    for (_, vec) in weights.iter().enumerate()  {
        for (uid, value ) in vec {
            sum_weights[*uid as usize] += value;
        }
    }
    println!("Weight For UID {:?} Sum {:?}", DEBUG_UID, sum_weights[DEBUG_UID]);
    println!("Weight Total Sum {:?}", sum_weights.iter().sum::<I32F32>());

    println!("-----------------Incentive-------------");        
    // let incentive = parse_incentive_from_file("debugging_logs/incentive.txt");
    let mut incentive = matmul_sparse(&weights, &active_stake, 256);
    inplace_normalize(&mut incentive); // range: I32F32(0, 1)
    println!("Incentive for UID {:?}: {:?}, Total Sum: {:?}", DEBUG_UID, incentive[DEBUG_UID], incentive.iter().sum::<I32F32>());
    // println!("{:?}", incentive);

    println!("-----------------BONDS DELTA-------------");
    // let bonds_delta = parse_bonds_from_file("debugging_logs/bonds_5_delta(norm).txt");
    let mut bonds_delta: Vec<Vec<(u16, I32F32)>> = row_hadamard_sparse(&weights, &active_stake);         
    inplace_col_normalize_sparse(&mut bonds_delta, 256); // sum_i b_ij = 1
    
    let mut sum_bonds_delta: Vec<I32F32> = vec![I32F32::from_num(0.0); bonds_delta.len()];
    for (_, vec) in bonds_delta.iter().enumerate() {
        for (uid, value) in vec {
            sum_bonds_delta[*uid as usize] += value;
        }
    }
    println!("Bonds Delta For UID {:?} Sum {:?}", DEBUG_UID, sum_bonds_delta[DEBUG_UID]);
    println!("Bonds Delta Total Sum {:?}", sum_bonds_delta.iter().sum::<I32F32>());

    println!("-----------------EMA BONDS-------------");
    let ema_bonds = parse_weights_from_file("debugging_logs/bonds_6_ema.txt");
    // println!("index:\t incentive\tdividands");
    let mut sum_ema_bonds = I32F32::from_num(0);
    let mut ema_bonds_idx_list = Vec::new();
    for (index, vec) in ema_bonds.iter().enumerate()  {
        // sum = I32F32::from_num(0);
        for (uid, value) in vec {            
            if *uid == DEBUG_UID as u16 {
                ema_bonds_idx_list.push(index);
                sum_ema_bonds += value;
                println!("EMA Bond for UID {:?} Index:{:?}  Value:{:?}", DEBUG_UID, index, value);
            }
        }       
        // println!("{:<10?}{:<15?}{:?}", index , incentive[index], dividends[index]);
    }
    println!("EMA Bonds Sum for UID {:?}: {:?}", DEBUG_UID, sum_ema_bonds);  

    println!("-----------------DIVIDENDS-------------");    
    let mut dividends_sum_included_uid: I32F32 = I32F32::from_num(0);
    // let dividends = parse_incentive_from_file("debugging_logs/dividends.txt");
    let mut dividends = matmul_transpose_sparse(&ema_bonds, &incentive);
    inplace_normalize(&mut dividends);    
    for index in ema_bonds_idx_list {
        dividends_sum_included_uid += dividends[index];
        println!("Dividend included UID  {:?} Index: {:?} , Value: {:?}", DEBUG_UID, index, dividends[index]);
    }        
    println!("Dividend Total Sum: {:?}, Sum Included UID {:?}: {:?}", dividends.iter().sum::<I32F32>(), DEBUG_UID, dividends_sum_included_uid);    
    println!("-----------------THE END-------------");

}

fn read_file_to_string(file_path: &str) -> io::Result<String> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}


fn parse_incentive_from_file(file_path: &str) -> Vec<I32F32> {

    let input = match read_file_to_string(file_path) {
        Ok(result) => result,
        Err(e) => {
            println!("Failed to read file: {:?}", e);
            return vec![];
        }
    };

    input.trim_matches(|c| c == '[' || c == ']')
         .split(',')
         .filter_map(|s| s.trim().parse::<I32F32>().ok())
         .collect()
}

fn parse_weights_from_file(file_path: &str) -> Vec<Vec<(u16, I32F32)>> {

    let input = match read_file_to_string(file_path) {
        Ok(result) => result,
        Err(e) => {
            println!("Failed to read file: {:?}", e);
            return vec![];
        }
    };

    let re = Regex::new(r"\((\d+),\s*([\d\.]+)\)").unwrap();

    // Create a vector to store the parsed data
    let mut data: Vec<Vec<(u16, I32F32)>> = Vec::new();

    // Split the input string into separate sub-lists
    let lists: Vec<&str> = input.trim_matches(|c| c == '[' || c == ']').split("], [").collect();

    for list in lists {
        let mut inner_vec: Vec<(u16, I32F32)> = Vec::new();
        for cap in re.captures_iter(list) {
            let int_part: u16 = cap[1].parse().unwrap();
            let float_part: I32F32 = cap[2].parse().unwrap();
            inner_vec.push((int_part, float_part));
        }
        data.push(inner_vec);
    }
    return data;
}

