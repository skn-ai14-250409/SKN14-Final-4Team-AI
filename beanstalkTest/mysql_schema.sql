-- Fixed MySQL schema (with commas).
SET NAMES utf8mb4;
SET time_zone = '+09:00';

-- Table `app_product`
DROP TABLE IF EXISTS `app_product`;
CREATE TABLE `app_product` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  `image_url` VARCHAR(200) NOT NULL,
  `spec` TEXT NOT NULL,
  `material` VARCHAR(255) NOT NULL,
  `esg_report_url` VARCHAR(200) NOT NULL,
  `external_id` VARCHAR(32) NULL,
  `url` VARCHAR(200) NOT NULL,
  `composition_parts` TEXT NULL,
  `details_bullets` TEXT NULL,
  `details_intro` TEXT NOT NULL,
  `fit` VARCHAR(64) NOT NULL,
  `impact` TEXT NULL,
  `made_in` VARCHAR(128) NOT NULL,
  `price` DOUBLE NULL,
  `sustainable_detail` TEXT NOT NULL,
  `sustainable_icons` TEXT NULL,
  `category` VARCHAR(64) NOT NULL,
  `color_detail` VARCHAR(64) NOT NULL,
  `color` VARCHAR(64) NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `app_product_external_id_ce7acc55` ON `app_product` (`external_id`);