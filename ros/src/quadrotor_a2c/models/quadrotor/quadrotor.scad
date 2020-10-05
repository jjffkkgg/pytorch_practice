length = 40;
edge = length*0.5;
r_d = edge - 1;
h_p = length*0.1*1.5;
r_p = length/20;
r_prop = 10;

module body(){
difference() { cube([length, length, length*0.08], center=true);
    translate([edge, edge, 0]) 
    cylinder(r=r_d, h=h_p, center=true);        translate([-edge, edge, 0]) 
    cylinder(r=r_d, h=h_p, center=true);      translate([edge, -edge, 0]) 
    cylinder(r=r_d, h=h_p, center=true);     translate([-edge, -edge, 0]) 
    cylinder(r=r_d, h=h_p, center=true); 
}
}
module pillar(){
translate([edge,0,0])
    cylinder(r=r_p,h=h_p, center=true);
translate([0,edge,0])
    cylinder(r=r_p,h=h_p, center=true);
translate([-edge,0,0])
    cylinder(r=r_p,h=h_p, center=true);
translate([0,-edge,0])
    cylinder(r=r_p,h=h_p, center=true);
}

module prop(){
translate([edge,0,h_p*0.5])
    cylinder(r=r_prop,h=r_prop*0.1, center=true);
translate([-edge,0,h_p*0.5])
    cylinder(r=r_prop,h=r_prop*0.1, center=true);
translate([0,edge,h_p*0.5])
    cylinder(r=r_prop,h=r_prop*0.1, center=true);
translate([0,-edge,h_p*0.5])
    cylinder(r=r_prop,h=r_prop*0.1, center=true);
}

module assembly(){ 

    body();
    pillar();
    prop();

}

assembly();