use substrate_api_client::rpc::JsonrpseeClient;
use substrate_api_client::{Api, GetStorage};
use substrate_api_client::ac_primitives::{AssetRuntimeConfig, Config};
use substrate_fixed::types::I32F32;

type AccountId = <AssetRuntimeConfig as Config>::AccountId;

// Get subnetwork size.
pub async fn get_subnetwork_n(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> u16 {

    let storage_value:Option<u16> = match api.get_storage_map ("SubtensorModule", "SubnetworkN", netuid, None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get subnetwork size: {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(0);
}
// Get current block number
pub async fn get_current_block_number(api: &Api<AssetRuntimeConfig, JsonrpseeClient>) -> u64 {
    
    let storage_value:Option<u32> = match api.get_storage("System", "Number", None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get current block number: {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(0) as u64;
}

// Get activity cutoff.
pub async fn get_activity_cutoff(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> u64 {

    let storage_value:Option<u16> = match api.get_storage_map ("SubtensorModule", "ActivityCutoff", netuid, None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get activity cutoff: {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(0) as u64;
}

// Get last update 
pub async fn get_last_update (api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> Vec<u64> {

    let storage_value:Option<Vec<u64>> = match api.get_storage_map ("SubtensorModule", "LastUpdate", netuid, None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get last update: {:?}", e);
            return vec![];
        }
    };

    return storage_value.unwrap_or(vec![]);

}

// Block at registration vector (block when each neuron was most recently registered
pub async fn get_block_at_registration(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16, n: u16) -> Vec<u64> {

    let mut block_at_registration: Vec<u64>  = Vec::new();
    for neuron_uid in 0..n {
        let reg_value = get_neuron_block_at_registration(api, netuid, neuron_uid).await;
        block_at_registration.push(reg_value);
    }   

    block_at_registration
}

pub async fn get_neuron_block_at_registration(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16, neuron_uid: u16) -> u64 {

    let storage_value:Option<u64> = match api.get_storage_double_map ("SubtensorModule", "BlockAtRegistration", netuid, neuron_uid, None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get block at registration: {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(0);

}

pub async fn get_keys(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> Vec<(u16, AccountId)> {

    let mut hotkeys: Vec<(u16, AccountId)> = Vec::new();
    let storage_double_map_key_prefix = api
                                                        .get_storage_double_map_key_prefix("SubtensorModule", "Keys", netuid)
                                                        .await
                                                        .unwrap();
    let double_map_storage_keys = api
                                                        .get_storage_keys_paged(Some(storage_double_map_key_prefix), 256, None, None)
                                                        .await
                                                        .unwrap();

    for (uid, storage_key) in double_map_storage_keys.iter().enumerate() {
        let storage_data:  AccountId = api.get_storage_by_key(storage_key.clone(), None).await.unwrap().unwrap();
        hotkeys.push((uid as u16, storage_data));
    }
    
    return hotkeys;
}

pub async fn get_total_stake_for_hotkey(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, hotkey: &AccountId) -> u64 {

    let storage_value:Option<u64> = match api.get_storage_map ("SubtensorModule", "TotalHotkeyStake", hotkey,  None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get total stake for hotkey: {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(0);

}

pub async fn get_validator_permit(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> Vec<bool> {

    let storage_value:Option<Vec<bool>> = match api.get_storage_map ("SubtensorModule", "ValidatorPermit", netuid,  None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get validator permit: {:?}", e);
            return vec![];
        }
    };

    return storage_value.unwrap_or(vec![]);

}

pub async fn get_max_allowed_validators(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> u16 {

    let storage_value:Option<u16> = match api.get_storage_map ("SubtensorModule", "MaxAllowedValidators", netuid,  None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get MaxAllowedValidators: {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(0);

}

pub async fn get_weights(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16, n: u16) -> Vec<Vec<(u16, I32F32)>> {

    let mut weights: Vec<Vec<(u16, I32F32)>> = vec![vec![]; n as usize];

    for uid_i in 0..n {

        let weights_i:  Option<Vec<(u16, u16)>> = match api.get_storage_double_map("SubtensorModule", "Weights", netuid, uid_i, None).await {
            Ok(result) => result,
            Err(e) => {
                log::error!("Can't get weights: {:?}", e);
                continue;
            }
        };
        
        for (uid_j, weight_ij) in weights_i.unwrap_or(vec![]).iter().filter(|(uid_j, _)| *uid_j < n ) {
            weights
                .get_mut(uid_i as usize)
                .expect("uid_i is filtered to be less than n; qed")
                .push((*uid_j, I32F32::from_num(*weight_ij)));
        }
    }
    
    return weights;

}

pub async fn get_bonds(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16, n: u16) -> Vec<Vec<(u16, I32F32)>> {

    let mut bonds: Vec<Vec<(u16, I32F32)>> = vec![vec![]; n as usize];

    for uid_i in 0..n {

        let bonds_vec:  Option<Vec<(u16, u16)>> = match api.get_storage_double_map("SubtensorModule", "Bonds", netuid, uid_i, None).await {
            Ok(result) => result,
            Err(e) => {
                log::error!("Can't get Bonds: {:?}", e);
                continue;
            }
        };

        for (uid_j, bonds_ij) in bonds_vec.unwrap_or(vec![]).iter().filter(|(uid_j, _)| *uid_j < n ) {
            bonds
                .get_mut(uid_i as usize)
                .expect("uid_i is filtered to be less than n; qed")
                .push((*uid_j, I32F32::from_num(*bonds_ij)));
        }
    }
    
    return bonds;

}

pub async fn get_kappa(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> u16 {

    let storage_value:Option<u16> = match api.get_storage_map ("SubtensorModule", "Kappa", netuid,  None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get Kappa: {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(0);

}

pub async fn get_liquid_alpha_on(api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> bool {

    let storage_value:Option<bool> = match api.get_storage_map ("SubtensorModule", "LiquidAlphaOn", netuid,  None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get LiquidAlphaOn {:?}", e);
            return false;
        }
    };

    return storage_value.unwrap_or(false);

}

pub async fn get_bonds_moving_average (api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> u64 {

    let storage_value:Option<u64> = match api.get_storage_map ("SubtensorModule", "BondsMovingAverage", netuid,  None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get BondsMovingAverage {:?}", e);
            return 0;
        }
    };

    return storage_value.unwrap_or(900000);
    
}

pub async fn get_alpha_values (api: &Api<AssetRuntimeConfig, JsonrpseeClient>, netuid: u16) -> (u16,u16) {

    let storage_value:Option<(u16, u16)> = match api.get_storage_map ("SubtensorModule", "AlphaValues", netuid,  None).await {
        Ok(result) => { result },
        Err(e) => {
            log::error!("Can't get AlphaValues {:?}", e);
            return (0,0);
        }
    };

    return storage_value.unwrap_or((0,0));

}

