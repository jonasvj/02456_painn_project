from src.data import MD17DataModule

def main():
    dm = MD17DataModule(
        molecule_name="ethanol",
        data_dir="data_md17",
        batch_size_train=32,
        batch_size_inference=64,
        num_workers=0,
        splits=(800, 100, 100),   # por ejemplo, para pruebas
        seed=0,
        subset_size=1000,         # usamos solo 1000 muestras para ir rápido
    )

    dm.prepare_data()
    dm.setup()

    print("Tamaños de los splits:")
    print("  train:", len(dm.data_train))
    print("  val:  ", len(dm.data_val))
    print("  test: ", len(dm.data_test))

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print("\nCampos del batch de train:")
    print("  z shape:     ", batch.z.shape)       # (N_átomos_total_del_batch,)
    print("  pos shape:   ", batch.pos.shape)     # (N_átomos_total_del_batch, 3)
    print("  energy shape:", batch.energy.shape)  # (N_moléculas_en_el_batch,)
    print("  force shape: ", batch.force.shape)   # (N_átomos_total_del_batch, 3)
    print("  batch shape: ", batch.batch.shape)   # mapea átomos → índice de molécula

if __name__ == "__main__":
    main()
