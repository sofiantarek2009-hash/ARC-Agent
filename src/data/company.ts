export const NORTH_STAR_DATA = {
  company: {
    name: 'NorthStar Energy Solutions',
    shortName: 'NorthStar',
    legalNote: 'Formerly Well Experts & Adriatic Engineering Solutions',
    contact: {
      hq: {
        label: 'Middle East HQ',
        address: 'Building 5027, Block 5, Meraj, Maadi, Cairo, Egypt',
        city: 'Cairo, Egypt',
        phones: ['+20 2 2082 6046', '+20 10 7078 7297'],
        email: 'info@well-experts.org',
      },
      aes: {
        label: 'Engineering & Inspection',
        city: 'Cairo, Egypt',
        email: 'info@adriatic-me.com',
        salesEmail: 'sales@adriatic-me.com',
      },
      fax: '+20 2 2447 2904',
      phone247: '+20 12 1200 0531',
    },
    established: 2016,
    tagline: 'Where Physical Engineering Meets Digital Simulation',
    profile: 'NorthStar Energy Solutions integrates the digital frontiers of drilling simulation with physical engineering inspection. Well Experts (est. 2022) is the first drilling and well control simulator manufacturer in the MENA region. Adriatic Engineering Solutions (est. 2016) provides Asset Integrity Management, NDT, lifting, and rope access services to the oilfield, marine, and industrial sectors.',
    vision: 'To set the global benchmark for digitally enabled well operations and industrial inspection, where integrated simulation, engineering, and competency ecosystems redefine safety, performance, and execution certainty across the energy sector.',
    mission: 'Banking on our people as our main asset, our mission is to provide reliable, innovative solutions that enable our customers to outperform safely across the Oil & Gas, marine, and industrial sectors.',
    stats: [
      { label: 'Clients Served', value: 34, suffix: '+' },
      { label: 'Industry Certifications', value: 9, suffix: '' },
      { label: 'Years in Field', value: 9, suffix: '+' },
      { label: 'Active Projects', value: 24, suffix: '+' },
    ],
    valueProps: [
      { title: 'MENA-Built Simulation', desc: 'The RDS-100® is the first drilling and well control simulator fully developed and manufactured in the Middle East and Africa.' },
      { title: 'Dual Accreditation', desc: 'IWCF Primary Accredited Training Company and official IADC WellSharp® Well Control Program provider.' },
      { title: 'Physical + Digital', desc: 'One entity combining rig inspection, NDT, and rope access with high-fidelity simulation and VR training.' },
      { title: 'Zero LTI Record', desc: 'Zero Lost Time Injuries across all field operations and training deployments.' },
    ],
  },
  rds100: {
    name: 'RDS-100® Reality Drilling Simulator',
    tagline: 'The first drilling and well control simulator fully developed in the MENA region.',
    formFactors: ['Full-Scale', 'Portable'],
    features: [
      'Live downhole pressure display during well control events — no pausing to pull up graphs',
      'Import real survey data for realistic torque and drag prediction',
      '15+ drilling problem scenarios that show realistically in surface parameters',
      'Real-time kill sheet calculations for tapered and deviated holes',
      'Complete stuck pipe module with realistic surface indicators',
      'Rig Sense: real-time drilling trend monitoring and alerting',
      'Continuous real-time data plotting for performance assessment',
    ],
    deployments: [
      { location: 'Cairo, Egypt', client: 'Well Experts HQ', year: 2022 },
      { location: 'Muscat, Oman', client: 'Ryboa Hima', year: 2024 },
    ],
  },
  milestones: [
    { date: '2016', title: 'Adriatic Engineering Solutions Founded', desc: 'Established in Cairo as a specialized lifting, inspection, and NDT services provider for the oilfield and marine sectors.' },
    { date: 'June 2022', title: 'Well Experts Founded', desc: 'Launched as the first drilling and well control simulator manufacturer in the Middle East and Africa.' },
    { date: 'Sept 2024', title: 'RDS-100 Deployed to Oman', desc: 'Successful deployment of the RDS-100 Advanced Drilling Simulator at Ryboa Hima in Muscat, Oman.' },
    { date: 'Jan 2025', title: 'IADC Membership', desc: 'Became a member of the International Association of Drilling Contractors, reinforcing global industry standards.' },
    { date: 'May 2025', title: 'IWCF Primary Accreditation', desc: 'Achieved IWCF Primary Accredited Training Company status for well control certification.' },
    { date: 'Jul 2025', title: 'IADC WellSharp® Provider', desc: 'Officially accredited to deliver IADC WellSharp® Well Control Accreditation Programs.' },
    { date: 'Nov 2025', title: 'Cheiron HSE Day', desc: 'Participated in Cheiron Petroleum\'s HSE Day, demonstrating RDS-100 simulation capabilities.' },
  ],
  services: {
    technology: {
      id: 'technology',
      name: 'Technology & Simulation',
      subcategories: [
        {
          name: 'Simulators Manufacturing (RDS-100®)',
          desc: 'High-fidelity drilling and well control simulation with live downhole pressure display, real survey data import, 15+ drilling problem scenarios, stuck pipe module, and real-time kill sheet calculations.',
        },
        {
          name: 'Virtual Reality (WEXPLORE®)',
          desc: 'Immersive, scenario-based VR environments exposing personnel to critical field risks with full spatial awareness before deployment to the rig floor.',
        },
        {
          name: 'Digital Solutions',
          desc: 'Software development, data analytics, machine learning, and AI-driven workflow automation for well operations.',
        },
        {
          name: 'Competency Assurance Programs (CAP)',
          desc: 'Independent, evidence-based assessment of critical well operations roles via high-stakes simulation and structured evaluation.',
        },
        {
          name: 'Integrated Project Management (IPM)',
          desc: 'End-to-end project lifecycle management for simulation deployment, training center commissioning, and digital transformation.',
        },
      ],
    },
    engineering: {
      id: 'engineering',
      name: 'Engineering & Inspection',
      subcategories: [
        {
          name: 'Rig Audits & Structural Inspection',
          desc: 'Comprehensive Rig Audits, structural fall protection surveys, Heavy Derrick Inspection per API 4G Cat III, DROPS surveys, and certified IRATA Rope Access services.',
        },
        {
          name: 'Asset Integrity Management',
          desc: 'Long-term asset integrity programs including preventive maintenance systems, calibrated inspection equipment, and compliance tracking.',
        },
        {
          name: 'Offshore & Marine Services',
          desc: 'Underwater Inspection in Lieu of Drydocking (UWILD), ultrasonic hull gauging per ABS/DNV Class rules, cathodic protection surveys, and marine lifting services.',
        },
        {
          name: 'Advanced NDT (PAUT/TOFD)',
          desc: 'Phased Array Ultrasonic Testing (PAUT), Time-of-Flight Diffraction (TOFD), and advanced techniques for critical weld and structural assessment.',
        },
        {
          name: 'Conventional NDT',
          desc: 'Magnetic Particle (MPI), Dye Penetrant (DPI), Ultrasonic (UT), Electromagnetic (EMI), Visual Testing (VT), and Eddy Current Testing (ECT).',
        },
        {
          name: 'Pressure Testing',
          desc: 'Hydrostatic and pneumatic pressure testing for pipelines, vessels, and process equipment to API and ASME standards.',
        },
        {
          name: 'Lifting Gear & Equipment',
          desc: 'Lifting gear inspection, wire rope and sling testing, crane load testing, Top Drive system engineering, and mud pump overhaul.',
        },
        {
          name: 'Equipment Maintenance',
          desc: 'Mechanical and electrical equipment maintenance, crane inspection per API RP 2D, forklift inspection, and fleet HSE assessments.',
        },
      ],
    },
    humanCapital: {
      id: 'human-capital',
      name: 'Training & Competency',
      subcategories: [
        {
          name: 'IWCF & IADC WellSharp® Training',
          desc: 'Primary IWCF Accredited Training Company and official IADC WellSharp® Well Control Accreditation Program provider.',
        },
        {
          name: 'Competency Assurance',
          desc: 'Independent assessment of Tool Pushers, Drillers, and Assistant Drillers via written exams, practical exams, on-the-job assessment, and simulation.',
        },
        {
          name: 'Technical Manpower',
          desc: 'Provision of Lifting Coaches, Certified API Inspectors, HSE Engineers, Industrial Divers, IRATA Rope Access Technicians, and qualified Welders.',
        },
        {
          name: 'Corporate Consultation',
          desc: 'HSE performance assessment via Persafe UK, Lean Six Sigma implementation, Asset Integrity Management, and organizational competency frameworks.',
        },
      ],
    },
    chemicals: {
      id: 'chemicals',
      name: 'Industrial Chemicals',
      desc: 'Optimized economic efficiency across chemical treatment formulations, raw industrial elements, and active intermediaries.',
      categories: [
        {
          name: 'Demulsifiers',
          products: [
            'NS-D100 Polymeric Surfactant',
            'NS-D250 Silicone-based Emulsion Breaker',
            'NS-D300 Phenolic Resin Blend',
            'NS-D450 Heavy Crude Treatment',
          ],
        },
        {
          name: 'Corrosion Inhibitors',
          products: [
            'NS-C10 Film-Forming Amine',
            'NS-C15 High-Temp Imidazoline',
            'NS-C40 Multiphase Pipeline Protector',
            'NS-C70 H2S/CO2 Neutralizer',
          ],
        },
        {
          name: 'Biocides',
          products: [
            'NS-B05 Glutaraldehyde 50%',
            'NS-B12 THPS Broad-Spectrum',
            'NS-B22 Quaternary Ammonium Blend',
          ],
        },
        {
          name: 'Scale Inhibitors',
          products: [
            'NS-S20 Phosphonate-Based Scale Blocker',
            'NS-S50 Polyacrylate Dispersant',
            'NS-S90 Barium/Strontium Specific',
          ],
        },
        {
          name: 'H2S Scavengers',
          products: [
            'NS-H2S Triazine-Based Liquid Scavenger',
            'NS-H2S Non-Triazine Fast-Acting',
            'NS-H2S Solid Bed Media',
          ],
        },
      ],
    },
  },
  // Separated into honest categories per the audit
  accreditations: [
    { name: 'IWCF', scope: 'Primary Accredited Training Company — well control certification', domain: 'iwcf.org', type: 'company' },
    { name: 'IADC', scope: 'WellSharp® Well Control Accreditation Program provider', domain: 'iadc.org', type: 'company' },
    { name: 'ISO 9001:2015', scope: 'Quality Management Systems — certified', domain: 'iso.org', type: 'company' },
  ],
  standardsWeInspectTo: [
    { name: 'API', scope: 'API 7K drilling equipment, API 4G Cat III derrick inspection, API RP 2D crane inspection', domain: 'api.org' },
    { name: 'ABS', scope: 'American Bureau of Shipping — marine and offshore structural surveys', domain: 'eagle.org' },
    { name: 'DNV', scope: 'Det Norske Veritas — marine classification and hull gauging surveys', domain: 'dnv.com' },
  ],
  personnelCertifications: [
    { name: 'ASNT', scope: 'NDT personnel qualification (Level II / III)', domain: 'asnt.org' },
    { name: 'AWS', scope: 'Welding procedure and personnel certification', domain: 'aws.org' },
    { name: 'AMPP', scope: 'Association for Materials Protection and Performance — corrosion control', domain: 'ampp.org' },
  ],
  memberships: [
    { name: 'LEEA', scope: 'Lifting Equipment Engineers Association — inspection standards', domain: 'leeaint.com' },
    { name: 'LIFT-TEX', scope: 'Synthetic lifting sling and webbing inspection', domain: 'lift-tex.co.uk' },
    { name: 'IRATA', scope: 'Industrial Rope Access Trade Association — technician certification', domain: 'irata.org' },
  ],
  vendorApprovals: [
    { name: 'Shell', scope: 'Approved simulator and well control training vendor', domain: 'shell.com' },
  ],
  // Flat list for backward compat with the marquee — combines all
  certifications: [
    { name: 'IWCF', scope: 'Primary Accredited Training Company', domain: 'iwcf.org' },
    { name: 'IADC', scope: 'WellSharp® Well Control Program provider', domain: 'iadc.org' },
    { name: 'ISO 9001:2015', scope: 'Quality Management Systems', domain: 'iso.org' },
    { name: 'API', scope: 'API 7K, API 4G Cat III, API RP 2D', domain: 'api.org' },
    { name: 'ABS', scope: 'Marine & offshore structural surveys', domain: 'eagle.org' },
    { name: 'DNV', scope: 'Marine classification & hull gauging', domain: 'dnv.com' },
    { name: 'ASNT', scope: 'NDT personnel Level II/III', domain: 'asnt.org' },
    { name: 'AWS', scope: 'Welding certification', domain: 'aws.org' },
    { name: 'AMPP', scope: 'Corrosion control standards', domain: 'ampp.org' },
    { name: 'LEEA', scope: 'Lifting gear inspection', domain: 'leeaint.com' },
    { name: 'LIFT-TEX', scope: 'Sling & webbing inspection', domain: 'lift-tex.co.uk' },
    { name: 'Shell', scope: 'Approved training vendor', domain: 'shell.com' },
  ],
  clients: [
    { name: 'BAPETCO', domain: 'shell.com.eg' },
    { name: 'BP', domain: 'bp.com' },
    { name: 'Enppi', domain: 'enppi.com' },
    { name: 'Cheiron', domain: 'cheironpetroleum.com' },
    { name: 'Ryboa Hima', domain: 'ryboahaima.com' },
    { name: 'Fos Energy', domain: 'fosenergy.com' },
    { name: 'Halliburton', domain: 'halliburton.com' },
    { name: 'Nabors', domain: 'nabors.com' },
    { name: 'Oxy', domain: 'oxy.com' },
    { name: 'Weatherford', domain: 'weatherford.com' },
    { name: 'SLB', domain: 'slb.com' },
    { name: 'Petrogulf Misr', domain: 'petrogulfmisr.com' },
    { name: 'EDC', domain: 'egyptian-drilling.com' },
    { name: 'PMS', domain: 'pms-eg.com' },
    { name: 'EMC', domain: 'emceg.com' },
    { name: 'Petrobel', domain: 'petrobel.org' },
    { name: 'EPRC', domain: 'eprc.gov.eg' },
    { name: 'Badr Petroleum', domain: 'bapetco.net' },
    { name: 'Eshpetco', domain: 'eshpetco.com' },
    { name: 'Qarun', domain: 'qarun.net' },
    { name: 'Khalda', domain: 'khalda-eg.com' },
    { name: 'GPC', domain: 'gpc.com.eg' },
    { name: 'Saknafta', domain: 'saknafta.com' },
    { name: 'Suez Steel Co.', domain: 'suezsteel.com' },
    { name: 'Saipem', domain: 'saipem.com' },
    { name: 'Petro Masila', domain: 'petromasila.com' },
    { name: 'Methanex', domain: 'methanex.com' },
    { name: 'ADES', domain: 'adesgroup.com' },
    { name: 'Gempco', domain: 'gempco.com' },
    { name: 'Maridive Group', domain: 'maridivegroup.net' },
    { name: 'Kuwait Petroleum Corp.', domain: 'kpc.com.kw' },
    { name: 'Expro', domain: 'expro.com' },
    { name: 'Sumed', domain: 'sumed.org' },
  ],
  partners: [
    { name: 'AME Offshore' },
    { name: 'APEX' },
    { name: 'EPHH' },
    { name: 'IGSAT' },
    { name: 'Marasi Marine' },
    { name: 'Persafe UK' },
  ],
};
