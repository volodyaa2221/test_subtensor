mod chain;
mod math;
use std::fs::File;
use std::io::{self, Read};


use regex::Regex;
use substrate_fixed::types::{I32F32, I64F64};

use substrate_api_client::rpc::JsonrpseeClient;
use substrate_api_client::Api;
use substrate_api_client::ac_primitives::{AssetRuntimeConfig, Config};


type AccountId = <AssetRuntimeConfig as Config>::AccountId;

use math::*;
use chain::*;

const DEBUG_UID: usize = 142;
const NET_UID: u16 = 8;

#[tokio::main]
async fn main() {

    env_logger::init();
    let client = JsonrpseeClient::with_default_url().await.unwrap();
	let api = Api::<AssetRuntimeConfig, _>::new(client).await.unwrap();
    let zero_num = I32F32::from_num(0.0);
    // Get subnetwork size.
    let n: u16 = get_subnetwork_n(&api, NET_UID).await;
    log::trace!("Number of Neurons in Network: {:?}", n);

    // ======================
    // == Active & updated ==
    // ======================

    // Get current block.
    let current_block: u64 = get_current_block_number(&api).await;
    log::trace!("current_block: {:?}", current_block);

    // Get activity cutoff.
    let activity_cutoff: u64 = get_activity_cutoff(&api, NET_UID).await;
    log::trace!("activity_cutoff: {:?}", activity_cutoff);

    // Last update vector.
    let last_update: Vec<u64> = get_last_update(&api, NET_UID).await;
    log::trace!("Last update: {:?}", &last_update);

    // Inactive mask.
    let inactive: Vec<bool> = last_update
        .iter()
        .map(|updated| updated.saturating_add(activity_cutoff) < current_block)
        .collect();
    log::trace!("Inactive: {:?}", inactive.clone());

    // Logical negation of inactive.
    let _active: Vec<bool> = inactive.iter().map(|&b| !b).collect();

    // Block at registration vector (block when each neuron was most recently registered).
    let block_at_registration: Vec<u64> = get_block_at_registration(&api, NET_UID, n).await;
    log::trace!("Block at registration: {:?}", &block_at_registration);

    // ===========
    // == Stake ==
    // ===========

    let hotkeys: Vec<(u16, AccountId)> = get_keys(&api, NET_UID).await;
    log::trace!("hotkeys: {:?}", &hotkeys);

    // Access network stake as normalized vector.
    let mut stake_64: Vec<I64F64> = vec![I64F64::from_num(0.0); n as usize];
    for (uid_i, hotkey) in &hotkeys {
        stake_64[*uid_i as usize] = I64F64::from_num(get_total_stake_for_hotkey(&api, hotkey).await);
    }
    log::trace!("Stake : {:?}", &stake_64);
    inplace_normalize_64(&mut stake_64);
    let stake: Vec<I32F32> = vec_fixed64_to_fixed32(stake_64);
    // range: I32F32(0, 1)
    log::trace!("Normalised Stake: {:?}", &stake);

    // =======================
    // == Validator permits ==
    // =======================

    // Get current validator permits.
    let validator_permits: Vec<bool> = get_validator_permit(&api, NET_UID).await;
    log::trace!("validator_permits: {:?}", validator_permits);

    // Logical negation of validator_permits.
    let validator_forbids: Vec<bool> = validator_permits.iter().map(|&b| !b).collect();

    // Get max allowed validators.
    let max_allowed_validators: u16 = get_max_allowed_validators(&api, NET_UID).await;
    log::trace!("max_allowed_validators: {:?}", max_allowed_validators);

    // Get new validator permits.
    let new_validator_permits: Vec<bool> = is_topk(&stake, max_allowed_validators as usize);
    log::trace!("new_validator_permits: {:?}", new_validator_permits);

    // ==================
    // == Active Stake ==
    // ==================

    let mut active_stake: Vec<I32F32> = stake.clone();

    // Remove inactive stake.
    inplace_mask_vector(&inactive, &mut active_stake);

    // Remove non-validator stake.
    inplace_mask_vector(&validator_forbids, &mut active_stake);

    // Normalize active stake.
    inplace_normalize(&mut active_stake);
    log::trace!("Active Stake:\n{:?}\n", &active_stake);

    // =============
    // == Weights ==
    // =============

    // Access network weights row unnormalized.
    let mut weights: Vec<Vec<(u16, I32F32)>> = get_weights(&api, NET_UID, n).await;
    log::trace!("Weights: {:?}", &weights);

    // Mask weights that are not from permitted validators.
    weights = mask_rows_sparse(&validator_forbids, &weights);
    log::trace!("Weights (permit): {:?}", &weights);

    // Remove self-weight by masking diagonal.
    weights = mask_diag_sparse(&weights);
    log::trace!("Weights (permit+diag): {:?}", &weights);

    // Remove weights referring to deregistered neurons.
    weights = vec_mask_sparse_matrix(
        &weights,
        &last_update,
        &block_at_registration,
        &|updated, registered| updated <= registered,
    );
    log::trace!("Weights (permit+diag+outdate): {:?}", &weights);

    // Normalize remaining weights.
    inplace_row_normalize_sparse(&mut weights);
    log::trace!("Weights (mask+norm): {:?}", &weights);

    // ================================
    // == Consensus, Validator Trust ==
    // ================================

    // Compute preranks: r_j = SUM(i) w_ij * s_i
    let preranks: Vec<I32F32> = matmul_sparse(&weights, &active_stake, n);
    log::trace!("Ranks (before): {:?}", &preranks);

    // Clip weights at majority consensus
    let kappa: I32F32 = get_float_kappa(&api, NET_UID).await; // consensus majority ratio, e.g. 51%.
    let consensus: Vec<I32F32> = weighted_median_col_sparse(&active_stake, &weights, n, kappa);
    log::trace!("Consensus: {:?}", &consensus);
/************************************************DEBUG***************************************************************************/
    let mut weights_list: Vec<I32F32> = vec![zero_num; n as usize];
    for (j, val) in &weights[DEBUG_UID] {
        weights_list[*j as usize] = *val;
    }

    log::debug!("Weights 142 ----- Consensus");
    for (index, value) in weights_list.iter().enumerate() {
        log::debug!("Weights[142] {:?} ----- {:?}", (index, value), consensus[index]);
    }

    log::debug!("Weight Sum: {:?} , Consensus Sum: {:?}", weights_list.iter().sum::<I32F32>(), consensus.iter().sum::<I32F32>());
 
/*********************************************************************************************************************************/    
    weights = col_clip_sparse(&weights, &consensus);
    log::trace!("Weights: {:?}", &weights);

/************************************************DEBUG***************************************************************************/
    

    for (j, val) in &weights[DEBUG_UID] {
        weights_list[*j as usize] = *val;
    }

    log::debug!("Weights 142 ----- Consensus");
    for (index, value) in weights_list.iter().enumerate() {
        log::debug!("Weights[142] {:?} ----- {:?}", (index, value), consensus[index]);
    }
    log::debug!("Weight Sum: {:?} , Consensus Sum: {:?}", weights_list.iter().sum::<I32F32>(), consensus.iter().sum::<I32F32>());

    // log::debug!("Weights list Index: {:?}", DEBUG_UID);
    // for vec in &weights[DEBUG_UID] {            
    //     log::debug!("{:?}", vec);        
    // }
    // log::debug!("SUM Weights Index {:?}: {:?}, Count:{:?}", DEBUG_UID, &weights[DEBUG_UID].iter().map(|v| v.1).sum::<I32F32>(), &weights[DEBUG_UID].len());

    // adjust_weights_for_uid(&api, NET_UID, DEBUG_UID, &mut weights, &active_stake, &last_update, &block_at_registration, &consensus, n).await;
    

    // for (index, val) in active_stake.iter().enumerate() {
    //     if *val > zero_num {
    //         log::debug!("Active Stake: {:?}", (index, val));
    //         // weights[index].push((DEBUG_UID as u16, weight_delta));
    //     }
    // }
    // log::debug!("Sum Active Stake: {:?}", active_stake.iter().sum::<I32F32>());
    // log::debug!("After adjust Weights list Index: {:?}", DEBUG_UID);
    // for vec in &weights[DEBUG_UID] {            
    //     log::debug!("{:?}", vec);        
    // }
    // log::debug!("SUM Weights Index {:?}: {:?}, Count:{:?}", DEBUG_UID, &weights[DEBUG_UID].iter().map(|v| v.1).sum::<I32F32>(), &weights[DEBUG_UID].len());
/*********************************************************************************************************************************/

    let validator_trust: Vec<I32F32> = row_sum_sparse(&weights);
    log::trace!("Validator Trust: {:?}", &validator_trust);

    // =============================
    // == Ranks, Trust, Incentive ==
    // =============================

    // Compute ranks: r_j = SUM(i) w_ij * s_i.
    let mut ranks: Vec<I32F32> = matmul_sparse(&weights, &active_stake, n);
    log::trace!("Ranks (after): {:?}", &ranks);

    // Compute server trust: ratio of rank after vs. rank before.
    let trust: Vec<I32F32> = vecdiv(&ranks, &preranks); // range: I32F32(0, 1)
    log::trace!("T: {:?}", &trust);

/************************************************DEBUG***************************************************************************/
    log::trace!("Ranks (after): {:?}", &ranks);
    log::trace!("Ranks (after) SUM: {:?}", &ranks.iter().sum::<I32F32>());
/*********************************************************************************************************************************/

    inplace_normalize(&mut ranks); // range: I32F32(0, 1)
    let incentive: Vec<I32F32> = ranks.clone();
    log::trace!("Incentive (=Rank): {:?}", &incentive);
/************************************************DEBUG***************************************************************************/
    log::debug!("Incentive: {:?}", &incentive);        
    log::debug!("Incentive SUM: {:?}, Count: {:?}", incentive.iter().sum::<I32F32>(), &incentive.clone().into_iter().filter(|v| *v > zero_num).collect::<Vec<I32F32>>().len());
/*********************************************************************************************************************************/
    // =========================
    // == Bonds and Dividends ==
    // =========================

    // Access network bonds.
    let mut bonds: Vec<Vec<(u16, I32F32)>> = get_bonds(&api, NET_UID, n).await;
    log::trace!("B: {:?}", &bonds);

    // Remove bonds referring to deregistered neurons.
    bonds = vec_mask_sparse_matrix(
        &bonds,
        &last_update,
        &block_at_registration,
        &|updated, registered| updated <= registered,
    );
    log::trace!("B (outdatedmask): {:?}", &bonds);

    // Normalize remaining bonds: sum_i b_ij = 1.
    inplace_col_normalize_sparse(&mut bonds, n);
    log::trace!("B (mask+norm): {:?}", &bonds);
/************************************************DEBUG***************************************************************************/
    // log::debug!("Bonds list Index : {:?}", DEBUG_UID);
    // for val in &bonds[DEBUG_UID] {            
    //     log::debug!("BONDS  {:?}", val);
    // }           
    // log::debug!("SUM Bonds Index {:?}: {:?}", DEBUG_UID, &bonds[DEBUG_UID].iter().map(|v| v.1).sum::<I32F32>());
/*********************************************************************************************************************************/

    // Compute bonds delta column normalized.
    let mut bonds_delta: Vec<Vec<(u16, I32F32)>> = row_hadamard_sparse(&weights, &active_stake); // ΔB = W◦S (outdated W masked)
    log::trace!("ΔB: {:?}", &bonds_delta);

    // Normalize bonds delta.
    inplace_col_normalize_sparse(&mut bonds_delta, n); // sum_i b_ij = 1
    log::trace!("ΔB (norm): {:?}", &bonds_delta);
/************************************************DEBUG***************************************************************************/
    // log::debug!("Bonds Delta list Index: {:?}", DEBUG_UID);
    // for (index, vec )in bonds_delta.iter().enumerate() {            
    //     for (uid, val) in vec {
    //         if *uid ==96 {
    //             log::debug!("BONDS DELTA  {:?}", (index, uid, val, active_stake[index]));
    //         }
    //     }
    // }           
    // log::debug!("SUM Bonds Delta Index {:?}: {:?}", DEBUG_UID, &bonds_delta[DEBUG_UID].iter().map(|v| v.1).sum::<I32F32>());
/*********************************************************************************************************************************/

    // Compute the Exponential Moving Average (EMA) of bonds.
    let mut ema_bonds =
            compute_ema_bonds_sparse(&api, NET_UID, consensus.clone(), bonds_delta, bonds).await;
    // Normalize EMA bonds.
    inplace_col_normalize_sparse(&mut ema_bonds, n); // sum_i b_ij = 1
    log::trace!("Exponential Moving Average Bonds: {:?}", &ema_bonds);
/************************************************DEBUG***************************************************************************/
    log::debug!("EMA bonds list Index: {:?} -- Incentive", DEBUG_UID);
    let mut sum_ema = zero_num;
    let mut sum_incentive = zero_num;
    let mut count = 0;
    for (uid, val) in &ema_bonds[DEBUG_UID] {            
        log::debug!("EMA BONDS ({:?}, {:?}) -- {:?}", uid, val, incentive[*uid as usize]);
        sum_ema += val;
        sum_incentive += incentive[*uid as usize];
        count +=1;
    }           
    log::debug!("Sum ema:{:?} , incentive:{:?}, count:{:?}", sum_ema, sum_incentive, count);
    log::debug!("SUM EMA bonds Index {:?}: {:?}", DEBUG_UID, &ema_bonds[DEBUG_UID].iter().map(|v| v.1).sum::<I32F32>());
/*********************************************************************************************************************************/
    // Compute dividends: d_i = SUM(j) b_ij * inc_j.
    // range: I32F32(0, 1)
    let mut dividends: Vec<I32F32> = matmul_transpose_sparse(&ema_bonds, &incentive);
    log::debug!("Dividends: {:?}", &dividends);
    log::debug!("Dividend: {:?}", &dividends[DEBUG_UID]);

    inplace_normalize(&mut dividends);
    log::debug!("Dividends: {:?}", &dividends);
    log::debug!("Dividend: {:?}", &dividends[DEBUG_UID]);

}

#[allow(dead_code)]
async fn adjust_weights_for_uid(
    api: &Api<AssetRuntimeConfig, JsonrpseeClient>, 
    netuid: u16,
    uid: usize, 
    weights:&mut Vec<Vec<(u16, I32F32)>>, 
    stake: &Vec<I32F32>,
    last_update: &[u64],
    block_at_registration: &[u64],    
    consensus: &Vec<I32F32>, 
    n: u16) {

    const TEST_UID: u16 = 22;
    let alpha = get_alpha_value(api, netuid, uid, consensus).await;
    let alpha_f64 = alpha.to_num::<f64>();
    
    let one_minus_alpha: I32F32 = I32F32::from_num(1.0).saturating_sub(alpha);
    let one_minus_alpha_f64 = one_minus_alpha.to_num::<f64>();

    let zero_num = I32F32::from_num(0.0);
        log::debug!("alpha, one minus alpha, {:?}", (alpha_f64, one_minus_alpha_f64));
    // for row in &mut *weights {                
    //         row.iter_mut()
    //             .for_each(|(_j, val)| *val = val.saturating_mul(I32F32::from_num(10000)))                
    // }    

    let mut incentive = matmul_sparse(&weights, &stake, n); 
    // let sum_rank = ranks.iter().sum::<I32F32>().to_num::<f64>();                
    inplace_normalize(&mut incentive); // range: I32F32(0, 1)
    log::debug!("{:?}", incentive);

    // Access network bonds.
    let mut bonds: Vec<Vec<(u16, I32F32)>> = get_bonds(&api, netuid, n).await;
    // Remove bonds referring to deregistered neurons.
    bonds = vec_mask_sparse_matrix(
        &bonds,
        last_update,
        block_at_registration,
        &|updated, registered| updated <= registered,
    );
    // Normalize remaining bonds: sum_i b_ij = 1.
    inplace_col_normalize_sparse(&mut bonds, n);

    let mut bonds_list:Vec<I32F32> = vec![zero_num; n as usize];
    for (j, val) in &bonds[uid] {
        bonds_list[*j as usize] = *val;
    }

    let bonds_delta: Vec<Vec<(u16, I32F32)>> = row_hadamard_sparse(&weights, &stake); // ΔB = W◦S (outdated W masked)    
    // inplace_col_normalize_sparse(&mut temp_delta, n); // sum_i b_ij = 1

    let temp_delta: Vec<Vec<(u16, I32F32)>> = row_hadamard_sparse(&weights, &stake); // ΔB = W◦S (outdated W masked)  
    let mut temp_sum: I32F32 = zero_num;
    for (index, vec )in temp_delta.iter().enumerate() {            
        for (j, val) in vec {
            if *j == TEST_UID {
                temp_sum = temp_sum.saturating_add(*val);
                log::debug!("BONDS DELTA  {:?}", (index, j, val, stake[index]));
            }
        }
    }           
    log::debug!("SUM  {:?}", (temp_sum));

    let mut weights_list: Vec<I32F32> = vec![zero_num; n as usize];
    for (j, val) in &weights[uid] {
        weights_list[*j as usize] = *val;
    }
    
     weights[uid].clear();

    let mut col_sum: Vec<I32F32> = vec![zero_num; n as usize]; // assume square matrix, rows=cols
    for sparse_row in bonds_delta.iter() {
        for (j, value) in sparse_row.iter() {
            col_sum[*j as usize] = col_sum[*j as usize].saturating_add(*value);
        }
    }
    let mut diff_stake_list: Vec<f64> = vec![0.0; n as usize];
    for (j, val) in weights_list.iter().enumerate() {
        
        let val_f64 = val.to_num::<f64>();
        let incentive_value = incentive[j].to_num::<f64>();
        // let rank_value = ranks[j].to_num::<f64>();
        let col_sum_value = col_sum[j].to_num::<f64>();
        let stake_value = stake[uid].to_num::<f64>();
        let bonds_value = bonds_list[j].to_num::<f64>();
        // if val_f64 > 0.0 {
            // let weight_value = (incentive_value * col_sum_value) /(stake_value * (1.0-incentive_value));
            // let weight_value = (incentive_value * col_sum_value) / stake_value ;
            let weight_value = col_sum_value * (incentive_value - one_minus_alpha_f64 *bonds_value) / alpha_f64*stake_value;     
            // let weight_for_uid_f64 = (incentive_f64 * bond_delta_sum_f64
            //     - (bond_f64 * one_minus_alpha_f64))
            //     / active_stake_f64
            //     * alpha_f64;
   
            diff_stake_list[j] =  (val_f64 - weight_value) * stake_value;
            weights[uid].push((j as u16, I32F32::from_num(weight_value)));
        // }

    }

    // let  use_stake: Vec<I32F32> = stake.iter().copied().filter(|&s| s > zero_num).collect();
    // let  count_stake = (use_stake.len()-1) as f64;

    // for (diff_index, diff_val) in diff_stake_list.iter().enumerate() {
    //         'w_loop: for w_index in 0..n as usize {
    //             if stake[w_index] > zero_num && w_index != uid{
    //                 for vec in weights[w_index].iter_mut() {
    //                     let new_val = vec.1.to_num::<f64>() + diff_val/stake[w_index].to_num::<f64>();
    //                     if vec.0 == diff_index as u16 && new_val > zero_num {
    //                         vec.1 = I32F32::from_num(new_val);
    //                         break 'w_loop;
    //                     }
    //                 }
    //             }
    //         }
    // }
            

    temp_sum = zero_num;
    let mut temp_delta: Vec<Vec<(u16, I32F32)>> = row_hadamard_sparse(&weights, &stake); // ΔB = W◦S (outdated W masked) 
        inplace_col_normalize_sparse(&mut temp_delta, n); // sum_i b_ij = 1
 
    for (index, vec )in temp_delta.iter().enumerate() {            
        for (j, val) in vec {
            if *j == TEST_UID {
                temp_sum = temp_sum.saturating_add(*val);
                log::debug!("BONDS DELTA  {:?}", (index, j, val, stake[index], bonds_list[*j as usize]));
            }
        }
    }           
    log::debug!("SUM  {:?}", (temp_sum));


    // let stake_num = stake[uid].to_bits();     
    // let sum_rank = ranks.iter().sum::<I32F32>().to_bits();         
    // let weight =   stake_num*sum_rank / (I32F32::from_num(1.0).to_bits()-stake_num);    
    // log::debug!("-----------------{:?} * {:?} = {:?} ",stake_num, sum_rank, weight);
    // log::debug!("-----------------{:?} * {:?} = {:?} ",I32F32::from_bits(stake_num), I32F32::from_bits(sum_rank), I32F32::from_bits(weight));
}

async fn get_float_kappa(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> I32F32 {
    I32F32::from_num(get_kappa(api, netuid).await).saturating_div(I32F32::from_num(u16::MAX))
}
/// Calculate the logistic function parameters 'a' and 'b' based on alpha and consensus values.
///
/// # Args:
/// * `alpha_high` - The high alpha value.
/// * `alpha_low` - The low alpha value.
/// * `consensus_high` - The high consensus value.
/// * `consensus_low` - The low consensus value.
///
/// # Returns:
/// A tuple containing the slope 'a' and intercept 'b' for the logistic function.
pub fn calculate_logistic_params(
    alpha_high: I32F32,
    alpha_low: I32F32,
    consensus_high: I32F32,
    consensus_low: I32F32,
) -> (I32F32, I32F32) {
    // log::trace!("alpha_high: {:?}", alpha_high);
    // log::trace!("alpha_low: {:?}", alpha_low);
    // log::trace!("consensus_high: {:?}", consensus_high);
    // log::trace!("consensus_low: {:?}", consensus_low);
    // Check for division by zero
    // extra caution to ensure we never divide by zero
    if consensus_high <= consensus_low || alpha_low == 0 || alpha_high == 0 {
        // Return 0 for both 'a' and 'b' when consensus values are equal
        return (I32F32::from_num(0.0), I32F32::from_num(0.0));
    }

    // Calculate the slope 'a' of the logistic function.
    // a = (ln((1 / alpha_high - 1)) - ln((1 / alpha_low - 1))) / (consensus_low - consensus_high)
    let a = (safe_ln(
        (I32F32::from_num(1.0).saturating_div(alpha_high))
            .saturating_sub(I32F32::from_num(1.0)),
    )
    .saturating_sub(safe_ln(
        (I32F32::from_num(1.0).saturating_div(alpha_low)).saturating_sub(I32F32::from_num(1.0)),
    )))
    .saturating_div(consensus_low.saturating_sub(consensus_high));
    // log::trace!("a: {:?}", a);

    // Calculate the intercept 'b' of the logistic function.
    // b = ln((1 / alpha_low - 1)) + a * consensus_low
    let b = safe_ln(
        (I32F32::from_num(1.0).saturating_div(alpha_low)).saturating_sub(I32F32::from_num(1.0)),
    )
    .saturating_add(a.saturating_mul(consensus_low));
    // log::trace!("b: {:?}", b);

    // Return the calculated slope 'a' and intercept 'b'.
    (a, b)
}

/// Compute the alpha values using the logistic function parameters 'a' and 'b'.
///
/// # Args:
/// * `consensus` - A vector of consensus values.
/// * `a` - The slope of the logistic function.
/// * `b` - The intercept of the logistic function.
///
/// # Returns:
/// A vector of computed alpha values.
pub fn compute_alpha_values(consensus: &[I32F32], a: I32F32, b: I32F32) -> Vec<I32F32> {
    // Compute the alpha values for each consensus value.
    let alpha: Vec<I32F32> = consensus
        .iter()
        .map(|c| {
            // Calculate the exponent value for the logistic function.
            // exp_val = exp(b - a * c)
            let exp_val = safe_exp(b.saturating_sub(a.saturating_mul(*c)));

            // Compute the alpha value using the logistic function formula.
            // alpha = 1 / (1 + exp_val)
            I32F32::from_num(1.0).saturating_div(I32F32::from_num(1.0).saturating_add(exp_val))
        })
        .collect();

    // Log the computed alpha values for debugging purposes.
    // log::trace!("alpha: {:?}", alpha);

    // Return the computed alpha values.
    alpha
}

/// Clamp the alpha values between alpha_high and alpha_low.
///
/// # Args:
/// * `alpha` - A vector of alpha values.
/// * `alpha_high` - The high alpha value.
/// * `alpha_low` - The low alpha value.
///
/// # Returns:
/// A vector of clamped alpha values.
pub fn clamp_alpha_values(
    alpha: Vec<I32F32>,
    alpha_high: I32F32,
    alpha_low: I32F32,
) -> Vec<I32F32> {
    let clamped_alpha: Vec<I32F32> = alpha
        .iter()
        .map(|a| {
            // First, clamp the value to ensure it does not exceed the upper bound (alpha_high).
            // If 'a' is greater than 'alpha_high', it will be set to 'alpha_high'.
            // If 'a' is less than or equal to 'alpha_high', it remains unchanged.
            let clamped_a = a
                .min(&alpha_high)
                // Next, clamp the value to ensure it does not go below the lower bound (alpha_low).
                // If the value (after the first clamping) is less than 'alpha_low', it will be set to 'alpha_low'.
                // If the value is greater than or equal to 'alpha_low', it remains unchanged.
                .max(&alpha_low);
            // Return the clamped value.
            *clamped_a
        })
        .collect();
    // log::trace!("alpha_clamped: {:?}", clamped_alpha);
    clamped_alpha
}

/// Compute the Exponential Moving Average (EMA) of bonds using the clamped alpha values for a sparse matrix.
///
/// # Args:
/// * `bonds_delta` - A vector of bond deltas.
/// * `bonds` - A vector of bonds.
/// * `alpha` - A vector of clamped alpha values.
///
/// # Returns:
/// A vector of EMA bonds.
pub fn compute_ema_bonds_with_liquid_alpha_sparse(
    bonds_delta: &[Vec<(u16, I32F32)>],
    bonds: &[Vec<(u16, I32F32)>],
    alpha: Vec<I32F32>,
) -> Vec<Vec<(u16, I32F32)>> {
    // Compute the Exponential Moving Average (EMA) of bonds using the provided clamped alpha values.
    let ema_bonds = mat_ema_alpha_vec_sparse(bonds_delta, bonds, &alpha);

    // Log the computed EMA bonds for debugging purposes.
    // log::trace!(
    //     "Exponential Moving Average Bonds Liquid Alpha: {:?}",
    //     ema_bonds
    // );

    // Return the computed EMA bonds.
    ema_bonds
}

/// Compute the Exponential Moving Average (EMA) of bonds using the clamped alpha values.
///
/// # Args:
/// * `bonds_delta` - A vector of bond deltas.
/// * `bonds` - A vector of bonds.
/// * `alpha` - A vector of clamped alpha values.
///
/// # Returns:
/// A vector of EMA bonds.
pub fn compute_ema_bonds_with_liquid_alpha(
    bonds_delta: &[Vec<I32F32>],
    bonds: &[Vec<I32F32>],
    alpha: Vec<I32F32>,
) -> Vec<Vec<I32F32>> {
    // Compute the Exponential Moving Average (EMA) of bonds using the provided clamped alpha values.
    let ema_bonds = mat_ema_alpha_vec(bonds_delta, bonds, &alpha);

    // Log the computed EMA bonds for debugging purposes.
    // log::trace!(
    //     "Exponential Moving Average Bonds Liquid Alpha: {:?}",
    //     ema_bonds
    // );

    // Return the computed EMA bonds.
    ema_bonds
}

/// Compute the Exponential Moving Average (EMA) of bonds using a normal alpha value for a sparse matrix.
///
/// # Args:
/// * `bonds_delta` - A vector of bond deltas.
/// * `bonds` - A vector of bonds.
/// * `netuid` - The network ID.
///
/// # Returns:
/// A vector of EMA bonds.
pub async fn compute_ema_bonds_normal_sparse(
    api: &Api<AssetRuntimeConfig, JsonrpseeClient>,
    bonds_delta: &[Vec<(u16, I32F32)>],
    bonds: &[Vec<(u16, I32F32)>],
    netuid: u16,
) -> Vec<Vec<(u16, I32F32)>> {
    // Retrieve the bonds moving average for the given network ID and scale it down.
    let bonds_moving_average: I64F64 = I64F64::from_num(get_bonds_moving_average(api, netuid).await)
        .saturating_div(I64F64::from_num(1_000_000));

    // Calculate the alpha value for the EMA calculation.
    // Alpha is derived by subtracting the scaled bonds moving average from 1.
    let alpha: I32F32 =
        I32F32::from_num(1).saturating_sub(I32F32::from_num(bonds_moving_average));

    // Compute the Exponential Moving Average (EMA) of bonds using the calculated alpha value.
    let ema_bonds = mat_ema_sparse(bonds_delta, bonds, alpha);

    // Log the computed EMA bonds for debugging purposes.
    // log::trace!("Exponential Moving Average Bonds Normal: {:?}", ema_bonds);

    // Return the computed EMA bonds.
    ema_bonds
}

/// Compute the Exponential Moving Average (EMA) of bonds using a normal alpha value.
///
/// # Args:
/// * `bonds_delta` - A vector of bond deltas.
/// * `bonds` - A vector of bonds.
/// * `netuid` - The network ID.
///
/// # Returns:
/// A vector of EMA bonds.
pub async fn compute_ema_bonds_normal(
    api: &Api<AssetRuntimeConfig, JsonrpseeClient>,
    bonds_delta: &[Vec<I32F32>],
    bonds: &[Vec<I32F32>],
    netuid: u16,
) -> Vec<Vec<I32F32>> {
    // Retrieve the bonds moving average for the given network ID and scale it down.
    let bonds_moving_average: I64F64 = I64F64::from_num(get_bonds_moving_average(api, netuid).await)
        .saturating_div(I64F64::from_num(1_000_000));

    // Calculate the alpha value for the EMA calculation.
    // Alpha is derived by subtracting the scaled bonds moving average from 1.
    let alpha: I32F32 =
        I32F32::from_num(1).saturating_sub(I32F32::from_num(bonds_moving_average));

    // Compute the Exponential Moving Average (EMA) of bonds using the calculated alpha value.
    let ema_bonds = mat_ema(bonds_delta, bonds, alpha);

    // Log the computed EMA bonds for debugging purposes.
    // log::trace!("Exponential Moving Average Bonds Normal: {:?}", ema_bonds);

    // Return the computed EMA bonds.
    ema_bonds
}

/// Compute the Exponential Moving Average (EMA) of bonds based on the Liquid Alpha setting for a sparse matrix.
///
/// # Args:
/// * `netuid` - The network ID.
/// * `consensus` - A vector of consensus values.
/// * `bonds_delta` - A vector of bond deltas.
/// * `bonds` - A vector of bonds.
///
/// # Returns:
/// A vector of EMA bonds.
pub async fn compute_ema_bonds_sparse(
    api: &Api<AssetRuntimeConfig, JsonrpseeClient>,
    netuid: u16,
    consensus: Vec<I32F32>,
    bonds_delta: Vec<Vec<(u16, I32F32)>>,
    bonds: Vec<Vec<(u16, I32F32)>>,
) -> Vec<Vec<(u16, I32F32)>> {
    // Check if Liquid Alpha is enabled, consensus is not empty, and contains non-zero values.
    // This way we avoid the quantil function panic.
    let is_liquid_alpha_on = get_liquid_alpha_on(api, netuid).await;
    if is_liquid_alpha_on
        && !consensus.is_empty()
        && consensus.iter().any(|&c| c != I32F32::from_num(0))
    {
        // Calculate the 75th percentile (high) and 25th percentile (low) of the consensus values.
        let consensus_high = quantile(&consensus, 0.75);
        let consensus_low = quantile(&consensus, 0.25);
        // Further check if the high and low consensus values meet the required conditions.
        if (consensus_high > consensus_low) || consensus_high != 0 || consensus_low < 0 {
            // if (consensus_high > consensus_low) || consensus_high != 0) || consensus_low != 0 {
            // if (consensus_high > consensus_low) || consensus_low != 0 {
            log::trace!("Using Liquid Alpha");

            // Get the high and low alpha values for the network.
            let (alpha_low, alpha_high): (I32F32, I32F32) = get_alpha_values_32(api, netuid).await;
            log::trace!("alpha_low: {:?} alpha_high: {:?}", alpha_low, alpha_high);

            // Calculate the logistic function parameters 'a' and 'b' based on alpha and consensus values.
            let (a, b) = calculate_logistic_params(
                alpha_high,
                alpha_low,
                consensus_high,
                consensus_low,
            );

            // Compute the alpha values using the logistic function parameters.
            let alpha = compute_alpha_values(&consensus, a, b);

            // Clamp the alpha values between alpha_high and alpha_low.
            let clamped_alpha = clamp_alpha_values(alpha, alpha_high, alpha_low);

            // Compute the Exponential Moving Average (EMA) of bonds using the clamped alpha values.
            compute_ema_bonds_with_liquid_alpha_sparse(
                &bonds_delta,
                &bonds,
                clamped_alpha,
            )
        } else {
            log::trace!("Using Bonds Moving Average");

            // Compute the EMA of bonds using a normal alpha value.
            compute_ema_bonds_normal_sparse(api,&bonds_delta, &bonds, netuid).await
        }
    } else {
        log::trace!("Using Bonds Moving Average");

        // Compute the EMA of bonds using a normal alpha value.
        compute_ema_bonds_normal_sparse(api,&bonds_delta, &bonds, netuid).await
    }
}

async fn get_alpha_values_32(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> (I32F32, I32F32) {
    let (alpha_low, alpha_high): (u16, u16) = get_alpha_values(api, netuid).await;
    let converted_low = I32F32::from_num(alpha_low).saturating_div(I32F32::from_num(u16::MAX));
    let converted_high =
        I32F32::from_num(alpha_high).saturating_div(I32F32::from_num(u16::MAX));

    (converted_low, converted_high)
}

#[allow(dead_code)]
fn read_file_to_string(file_path: &str) -> io::Result<String> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

#[allow(dead_code)]
fn parse_incentive_from_file(file_path: &str) -> Vec<I32F32> {

    let input = match read_file_to_string(file_path) {
        Ok(result) => result,
        Err(e) => {
            log::error!("Failed to read file: {:?}", e);
            return vec![];
        }
    };

    input.trim_matches(|c| c == '[' || c == ']')
         .split(',')
         .filter_map(|s| s.trim().parse::<I32F32>().ok())
         .collect()
}

#[allow(dead_code)]
fn parse_weights_from_file(file_path: &str) -> Vec<Vec<(u16, I32F32)>> {

    let input = match read_file_to_string(file_path) {
        Ok(result) => result,
        Err(e) => {
            log::error!("Failed to read file: {:?}", e);
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

#[allow(dead_code)]
async fn get_alpha_value(
    api: &Api<AssetRuntimeConfig, JsonrpseeClient>, 
    netuid: u16, 
    uid: usize, 
    consensus: &Vec<I32F32> ) -> I32F32 {
    let is_liquid_alpha_on = get_liquid_alpha_on(api, netuid).await;
    if is_liquid_alpha_on
        && !consensus.is_empty()
        && consensus.iter().any(|&c| c != I32F32::from_num(0))
    {
        // Calculate the 75th percentile (high) and 25th percentile (low) of the consensus values.
        let consensus_high = quantile(consensus, 0.75);
        let consensus_low = quantile(consensus, 0.25);
        // Further check if the high and low consensus values meet the required conditions.
        if (consensus_high > consensus_low) || consensus_high != 0 || consensus_low < 0 {
            // if (consensus_high > consensus_low) || consensus_high != 0) || consensus_low != 0 {
            // if (consensus_high > consensus_low) || consensus_low != 0 {
            log::trace!("Using Liquid Alpha");

            // Get the high and low alpha values for the network.
            let (alpha_low, alpha_high): (I32F32, I32F32) = get_alpha_values_32(api, netuid).await;
            log::trace!("alpha_low: {:?} alpha_high: {:?}", alpha_low, alpha_high);

            // Calculate the logistic function parameters 'a' and 'b' based on alpha and consensus values.
            let (a, b) = calculate_logistic_params(
                alpha_high,
                alpha_low,
                consensus_high,
                consensus_low,
            );

            // Compute the alpha values using the logistic function parameters.
            let alpha = compute_alpha_values(&consensus, a, b);

            // Clamp the alpha values between alpha_high and alpha_low.
            let clamped_alpha = clamp_alpha_values(alpha, alpha_high, alpha_low);

            // Compute the Exponential Moving Average (EMA) of bonds using the clamped alpha values.
            return clamped_alpha[uid];
        } else {
            log::trace!("Using Bonds Moving Average");
            // Retrieve the bonds moving average for the given network ID and scale it down.
            let bonds_moving_average: I64F64 = I64F64::from_num(get_bonds_moving_average(api, netuid).await)
                .saturating_div(I64F64::from_num(1_000_000));

            // Calculate the alpha value for the EMA calculation.
            // Alpha is derived by subtracting the scaled bonds moving average from 1.            
            I32F32::from_num(1).saturating_sub(I32F32::from_num(bonds_moving_average))            
        }
    } else {

        log::trace!("Using Bonds Moving Average");

        let bonds_moving_average: I64F64 = I64F64::from_num(get_bonds_moving_average(api, netuid).await)
        .saturating_div(I64F64::from_num(1_000_000));
        // Calculate the alpha value for the EMA calculation.
        // Alpha is derived by subtracting the scaled bonds moving average from 1.
        I32F32::from_num(1).saturating_sub(I32F32::from_num(bonds_moving_average))

    }

}