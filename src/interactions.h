#pragma once

#include "intersections.h"
#include <math.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
glm::vec3 normal, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 lambertBxdf(glm::vec3 in, glm::vec3 out, const Material &m){
	return m.color / PI;
}

__host__ __device__
float lambertPDF(glm::vec3 in, glm::vec3 out, glm::vec3 normal){
	return glm::dot(glm::normalize(normal), in) / PI;
}

__host__ __device__
glm::vec3 perfectSpecularBxdf(glm::vec3 in, glm::vec3 normal, const Material &m){
	return m.specular.color / glm::abs(glm::dot(in, normal));
}

__host__ __device__
float perfectSpecularPDF(glm::vec3 in, glm::vec3 out, glm::vec3 normal){
	return 0.f;
}

__host__ __device__
glm::vec3 fresnelDielectricBxdf(glm::vec3 in, glm::vec3 out, glm::vec3 normal, const Material &m) {
	float etat, etai;
	glm::dot(in, normal) > 0.f ? etat = 1.f, etai = m.indexOfRefraction : etai = 1.f, etat = m.indexOfRefraction;

	float cost = glm::abs(glm::dot(in, normal));
	float cosi = glm::abs(glm::dot(out, normal));

	float Rparl = ((etat * cosi) - (etai * cost)) /
		((etat * cosi) + (etai * cost));
	float Rperp = ((etai * cosi) - (etat * cost)) /
		((etai * cosi) + (etat * cost));
	
	float fresnel = (Rparl*Rparl + Rperp*Rperp) / 2.f;
	return (etat * etat) / (etai * etai) * (1 - fresnel) * m.color / glm::abs(glm::dot(in, normal));
}

__host__ __device__
glm::vec3 fresnelConductorBxdf(glm::vec3 in, glm::vec3 normal, const Material &m){
	float cosi = glm::abs(glm::dot(in, normal));
	float tmp = (m.indexOfRefraction * m.indexOfRefraction + m.absorption * m.absorption) * cosi * cosi;
	float Rparl2 = (tmp - (2.f * m.indexOfRefraction * cosi) + 1) / (tmp + (2.f * m.indexOfRefraction * cosi) + 1);

	tmp = m.indexOfRefraction * m.indexOfRefraction + m.absorption * m.absorption;
	float Rperp2 = (tmp - (2.f * m.indexOfRefraction * cosi) + cosi*cosi) / (tmp + (2.f * m.indexOfRefraction * cosi) + cosi*cosi);
	return (Rparl2 + Rperp2) / 2.f * m.color / glm::abs(glm::dot(in, normal));
}

__host__ __device__
void sampleBxdf(
glm::vec3 &bxdf
, float &pdf
, glm::vec3 in
, glm::vec3 out
, glm::vec3 normal
, const Material &m
, thrust::default_random_engine &rng
, int bxdfContrib) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float matType = m.hasReflective + m.hasRefractive;
	if (matType < 0.01f) matType = 1.f;

	float r = u01(rng);
	if (r < m.hasReflective / matType) {
		bxdf = bxdfContrib > 0 ? fresnelConductorBxdf(in, normal, m) : glm::vec3(0.f);
		pdf = bxdfContrib > 0 ? 1.f : perfectSpecularPDF(in, out, normal);
	}
	else if (r < (m.hasRefractive + m.hasReflective) / matType) {
		bxdf = bxdfContrib > 0 ? fresnelDielectricBxdf(in, out, normal, m) : glm::vec3(0.f);
		pdf = bxdfContrib > 0 ? 1.f : 0.f;
	}
	else {
		bxdf = lambertBxdf(in, out, m);
		pdf = lambertPDF(in, out, normal);
	}

}

__host__ __device__
float lightPDF(glm::vec3 in, glm::vec3 normal, glm::vec3 intersect, glm::vec3 origin, float lightArea) {
	float cosTheta = glm::dot(-in, glm::normalize(normal));
	return glm::length(intersect - origin) / (cosTheta * lightArea);
}

__host__ __device__
glm::vec3 lightEnergy(glm::vec3 in, glm::vec3 normal, const Material & lm) {
	return glm::dot(-in, normal) > 0 ? lm.color * lm.emittance : glm::vec3(0.f);
}

__host__ __device__
float powerHeuristic(float pdf1, float pdf2){
	return (pdf1 * pdf2) / (pdf1 * pdf1 + pdf2 * pdf2);
}

__host__ __device__
void sampleBrdf(
PathSegment & pathSeg,
glm::vec3 intersect,
glm::vec3 normal,
const Material &m,
thrust::default_random_engine &rng
){
	if (m.hasReflective > 0.f) {
		pathSeg.ray.direction = glm::reflect(pathSeg.ray.direction, normal);
	}
	else if (m.hasRefractive > 0.f) {
		float IOR = pathSeg.inside ? m.indexOfRefraction : 1.f / m.indexOfRefraction;
		pathSeg.inside = !pathSeg.inside;
		pathSeg.ray.direction = glm::refract(pathSeg.ray.direction, normal, IOR);
	}
	else{
		pathSeg.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
	}

	pathSeg.ray.origin = intersect + (0.01f * pathSeg.ray.direction);
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
PathSegment & pathSegment,
glm::vec3 intersect,
glm::vec3 normal,
const Material &m,
thrust::default_random_engine &rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.
	if (m.hasReflective > 0.f) {
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	}
	else if (m.hasRefractive > 0.f) {
		float IOR = pathSegment.inside ? m.indexOfRefraction : 1.f / m.indexOfRefraction;
		pathSegment.inside = !pathSegment.inside;
		pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, IOR);
	}
	else{
		pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
	}
	pathSegment.color *= m.color;
	pathSegment.ray.origin = intersect + (0.01f * pathSegment.ray.direction);
	pathSegment.remainingBounces--;
}

__host__ __device__
void scatterWithDirectLighting(
PathSegment & pathSegment,
PathSegment & shadowFeeler,
PathSegment & brdfSample,
ShadeableIntersection & objIntersect,
ShadeableIntersection & lightIntersect,
ShadeableIntersection & brdfIntersect,
const Material &m,
const Material &lm,
const Material &bm,
int num_lights,
thrust::default_random_engine &rng){

	// Temp color
	glm::vec3 col;

	// Don't really do anything for refractive item?

	if (true) {
		// Direct lighting part
		glm::vec3 wi = glm::normalize(shadowFeeler.ray.direction);
		glm::vec3 lightContribution;
		glm::vec3 brdfContribution;
		float lightPdf = -0.001f;
		float brdfPdf = -0.001f;


		// We actually hit a light with our shadow feeler! Here we calculate the lighting sample.
		if (lightIntersect.t > 0.f && lm.emittance > 0.f) {

			lightContribution = lightEnergy(wi, lightIntersect.surfaceNormal, lm);
			lightPdf = lightPDF(wi, lightIntersect.surfaceNormal, lightIntersect.intersect,
				shadowFeeler.ray.origin, lightIntersect.surfaceArea);

			sampleBxdf(brdfContribution, brdfPdf, wi, pathSegment.ray.direction
				, objIntersect.surfaceNormal, m, rng, 0.f);

			if (lightPdf > 0.f && glm::length(lightContribution) > 0.f && brdfPdf > 0.f && glm::length(brdfContribution) > 0.f) {
				float dot_pdf = glm::abs(glm::dot(-wi, glm::normalize(objIntersect.surfaceNormal))) / lightPdf;
				float w = powerHeuristic(lightPdf, brdfPdf);
				col += w * brdfContribution * lightContribution * dot_pdf * (float)num_lights;
			}
		}
		else {
			col = glm::vec3(0.f);
		}

		// Reset variables
		wi = glm::vec3(0.f);
		brdfContribution = glm::vec3(0.f);
		lightContribution = glm::vec3(0.f);
		lightPdf = 0.f;
		brdfPdf = 0.f;

		// Compute brdf contribution
		wi = glm::normalize(brdfSample.ray.direction);
		sampleBxdf(brdfContribution, brdfPdf, wi, pathSegment.ray.direction
			, objIntersect.surfaceNormal, m, rng, 1.f);

		if (brdfPdf > 0.f && glm::length(brdfContribution) > 0.f) {
			// Brdf sample hit a light
			if (brdfIntersect.t > 0.f && bm.emittance > 0.f) {
				lightContribution = lightEnergy(wi, brdfIntersect.surfaceNormal, bm);
				lightPdf = lightPDF(wi, brdfIntersect.surfaceNormal, brdfIntersect.intersect, brdfSample.ray.origin, brdfIntersect.surfaceArea);

				if (lightPdf > 0.f && glm::length(lightContribution) > 0.f) {
					float dot_pdf = glm::max(0.f, glm::abs(glm::dot(wi, glm::normalize(objIntersect.surfaceNormal)))) / brdfPdf;
					float w = powerHeuristic(brdfPdf, lightPdf);
					col += w * brdfContribution * lightContribution * dot_pdf;
				}
			}
		}

		// Add new colors
		pathSegment.color += pathSegment.throughput * col;

		// Early exit
		if (glm::length(brdfContribution) <= 0.f || brdfPdf <= 0.f) {
			pathSegment.remainingBounces = 0;
		}

		// Update throughput
		brdfContribution *= glm::abs(glm::dot(wi, objIntersect.surfaceNormal)) / (brdfPdf * 2.f);
		pathSegment.throughput *= brdfContribution;
	}
	else {
		pathSegment.throughput *= m.color;
	}

	// Set up new path segments
	sampleBrdf(pathSegment,
		objIntersect.intersect,
		objIntersect.surfaceNormal,
		m,
		rng);
	pathSegment.remainingBounces--;

	// Russian Roulette!
	if (pathSegment.remainingBounces == 0) {
		thrust::uniform_real_distribution<float> u01(0, 1);
		if (u01(rng) > glm::min(0.5f, glm::length(pathSegment.throughput)))
			pathSegment.remainingBounces = 0;
	}
}

__host__ __device__
glm::vec3 sampleCube(thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	int face = (int)(5.95f * u01(rng));

	float p1 = u01(rng) - 0.5f;
	float p2 = u01(rng) - 0.5f;
	glm::vec3 P(0.f);

	// 0->1 is + and 1->2 is - on f1, 0->1 is x, 1->2 is y, 2->3 is z on f2
	if (face == 0){
		// +X
		P[0] = 0.5f;
		P[1] = p1;
		P[2] = p2;
	}
	else if (face == 1){
		// -X
		P[0] = -0.5f;
		P[1] = p1;
		P[2] = p2;
	}
	else if (face == 2){
		// +Y
		P[0] = p1;
		P[1] = 0.5f;
		P[2] = p2;
	}
	else if (face == 3){
		// -Y
		P[0] = p1;
		P[1] = -0.5f;
		P[2] = p2;
	}
	else if (face == 4){
		// +Z
		P[0] = p1;
		P[1] = p2;
		P[2] = 0.5f;
	}
	else {
		// -Z
		P[0] = p1;
		P[1] = p2;
		P[2] = -0.5f;
	}

	return P;
}

__host__ __device__
glm::vec3 sampleSphere(thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float z = 1.f - 2.f * u01(rng);
	float r = glm::sqrt(glm::max(0.f, 1.f - z*z));
	float phi = 2.f * PI * u01(rng);
	float x = r * glm::cos(phi);
	float y = r * glm::sin(phi);

	return glm::vec3(x / 2, y / 2, z / 2);
}