"""
Script to compute jet observables from reconstructed and truth jet images
using a SwinIR model for image reconstruction and FastJet for jet clustering.
"""

import numpy as np
import torch
from tqdm import tqdm
import fastjet as fj

from model import SwinIR as net
from torch.utils.data import DataLoader
from itertools import combinations

from config import BATCH_SIZE, MODEL_CONFIG


def extract_particles_from_image(jet_image, eta_range=(-3.0, 3.0), phi_range=(-np.pi, np.pi)):
    """
    Extract particles' (pt, eta, phi) from a jet image.

    Args:
        jet_image (np.ndarray): 2D array of pixel intensities representing pT.
        eta_range (tuple): (min_eta, max_eta) range of eta values.
        phi_range (tuple): (min_phi, max_phi) range of phi values.

    Returns:
        List of tuples: Each tuple is (pt, eta, phi) for non-zero pixels.
    """
    eta_bins, phi_bins = jet_image.shape
    eta_edges = np.linspace(eta_range[0], eta_range[1], eta_bins + 1)
    phi_edges = np.linspace(phi_range[0], phi_range[1], phi_bins + 1)
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    nonzero_indices = np.nonzero(jet_image)
    particles = []
    for eta_idx, phi_idx in zip(nonzero_indices[0], nonzero_indices[1]):
        pt = jet_image[eta_idx, phi_idx]
        eta = eta_centers[eta_idx]
        phi = phi_centers[phi_idx]
        particles.append((pt, eta, phi))
    return particles


def particle_to_pseudojet(particles, mass=0.1395):
    """
    Convert a list of particles to FastJet PseudoJet objects.

    Args:
        particles: List of (pt, eta, phi) tuples.
        mass: Assumed pion mass for each particle [GeV].

    Returns:
        List of fj.PseudoJet instances.
    """
    pseudojets = []
    for pt, eta, phi in particles:
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
        pseudojets.append(fj.PseudoJet(px, py, pz, E))
    return pseudojets


def compute_jet_girth(pseudojet):
    """
    Compute the girth of a jet.

    Girth = sum(pt_i * DeltaR(i, jet)) / pt_jet over all constituents.

    Args:
        pseudojet: FastJet PseudoJet object.

    Returns:
        Girth value (dimensionless).
    """
    constituents = pseudojet.constituents()
    if not constituents:
        return 0.0
    pt_jet = pseudojet.pt()
    if pt_jet == 0.0:
        return 0.0
    girth = sum(c.pt() * pseudojet.delta_R(c) for c in constituents) / pt_jet
    return girth


def compute_ECFS(jet_list, beta=2.0, theta_max=0.4):
    """
    Compute the Energy Correlation Function (ECF) for a list of jets.

    Only pairs with angular separation <= theta_max are considered.

    Args:
        jet_list: List of jets, each as a list of PseudoJet constituents.
        beta: Exponent for angular weighting.
        theta_max: Maximum angular separation for pairs.

    Returns:
        NumPy array of ECF sums for each jet.
    """
    ecf_values = []
    for constituents in tqdm(jet_list, desc="Computing ECFs"):
        if len(constituents) < 2:
            continue
        ecf_sum = 0.0
        for p1, p2 in combinations(constituents, 2):
            theta = p1.delta_R(p2)
            if theta <= theta_max:
                ecf_sum += (p1.pt() * p2.pt()) * (theta ** beta)
        ecf_values.append(ecf_sum)
    return np.array(ecf_values)


def process_jets(particle_list, jet_def):
    """
    Cluster particles into jets and compute basic observables.

    Args:
        particle_list: List of particle lists.
        jet_def: FastJet JetDefinition object.

    Returns:
        Tuple of (pt_array, mass_array, girth_array, list_of_jets).
    """
    pt_list, mass_list, girth_list = [], [], []
    jets_list = []
    for particles in tqdm(particle_list, desc="Clustering jets"):
        pseudojets = particle_to_pseudojet(particles)
        cs = fj.ClusterSequence(pseudojets, jet_def)
        jets = cs.inclusive_jets(ptmin=100.0)
        for jet in jets:
            pt_list.append(jet.pt())
            mass_list.append(jet.m())
            girth_list.append(compute_jet_girth(jet))
            jets_list.append(jet.constituents())
    return np.array(pt_list), np.array(mass_list), np.array(girth_list), jets_list


def main():
    """
    Main function to load datasets, run inference, extract particles, cluster jets, and save observables.
    """
    print("Loading datasets...")
    test_dataset = torch.load('datasets/test.pt', weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Dataloaders created.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        print("No GPUs available.")

    print("Instantiating model...")
    model = net(**MODEL_CONFIG).to(device)
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs.")
        model = torch.nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")

    state_dict = torch.load('models/best_model')
    model.load_state_dict(state_dict)
    print("Model instantiated.")

    # Perform inference and extract particles
    truth_particles, reco_particles = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Model inference"):
            model_inputs, truth_images = batch
            model_inputs = model_inputs.to(device)
            outputs = model(model_inputs)
            outputs = torch.clamp(outputs, min=0).cpu().numpy()
            truth_images = truth_images.numpy()

            for out_img, truth_img in zip(outputs, truth_images):
                out_img = out_img[0]
                truth_img = truth_img[0]
                out_img[out_img < 0.3] = 0.0
                truth_img[truth_img < 0.3] = 0.0
                reco_particles.append(extract_particles_from_image(out_img))
                truth_particles.append(extract_particles_from_image(truth_img))

    R = 0.4
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)

    truth_pt, truth_mass, truth_girth, truth_jets = process_jets(truth_particles, jet_def)
    reco_pt, reco_mass, reco_girth, reco_jets = process_jets(reco_particles, jet_def)

    truth_ecf = compute_ECFS(truth_jets)
    reco_ecf = compute_ECFS(reco_jets)

    # Save observables
    obs_map = {
        "truth_pt": truth_pt,
        "truth_mass": truth_mass,
        "truth_girth": truth_girth,
        "truth_ecf": truth_ecf,
        "reco_pt": reco_pt,
        "reco_mass": reco_mass,
        "reco_girth": reco_girth,
        "reco_ecf": reco_ecf,
    }
    for name, data in obs_map.items():
        np.save(f"obs/{name}", data)
        print(f"Saved {name} to obs/{name}.npy")


if __name__ == "__main__":
    main()

